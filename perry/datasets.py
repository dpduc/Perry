"""Datasets: CSV files and spreadsheet sheets stored as tables in the analytics schema.

Uploading writes through the main connection. Querying (`run_query`) connects as the
read-only analyst role, so whatever SQL the Analysis Agent writes, Postgres only lets
it read Datasets, within a statement timeout, and Perry returns at most a row limit.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from perry import db

SUPPORTED_SUFFIXES = {".csv", ".xlsx"}
SCHEMA = "analytics"

STATEMENT_TIMEOUT_MS = int(os.getenv("PERRY_ANALYSIS_TIMEOUT_MS", "10000"))
ROW_LIMIT = int(os.getenv("PERRY_ANALYSIS_ROW_LIMIT", "200"))

# Postgres identifiers are at most 63 bytes.
_MAX_IDENTIFIER = 63


@dataclass(frozen=True)
class Dataset:
    name: str  # also the table name the SQL uses
    origin: str  # the uploaded file it came from
    columns: list[str]
    types: list[str]
    row_count: int

    def describe(self) -> str:
        cols = ", ".join(f"{c} ({t})" for c, t in zip(self.columns, self.types, strict=True))
        return f"{self.name} ({self.row_count} rows, from {self.origin}): {cols}"


@dataclass(frozen=True)
class UploadResult:
    dataset: Dataset
    replaced: bool

    def describe(self) -> str:
        d = self.dataset
        verb = "replaced" if self.replaced else "created"
        return f"{d.name}: {verb}, {d.row_count} rows, columns {', '.join(d.columns)}"


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool  # more rows existed than the row limit


class QueryError(Exception):
    """A query that failed or timed out, with a message fit to show the user."""


def identifier(text: str, fallback: str) -> str:
    """A plain lowercase SQL identifier, so the model can write it without quoting."""
    name = re.sub(r"\W+", "_", text.lower()).strip("_") or fallback
    if name[0].isdigit():
        name = f"{fallback}_{name}"
    return name[:_MAX_IDENTIFIER]


def _read(path: Path) -> dict[str, pd.DataFrame]:
    """Dataset name -> its rows: one per CSV, one per spreadsheet sheet."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return {identifier(path.stem, "dataset"): pd.read_csv(path)}
    if suffix == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None)
        return {identifier(f"{path.stem} {sheet}", "dataset"): f for sheet, f in sheets.items()}
    raise ValueError(f"Unsupported file type: {path.name} (use .csv or .xlsx)")


def _column_names(labels) -> list[str]:
    names: list[str] = []
    for label in labels:
        base = identifier(str(label), "column")
        name, n = base, 2
        while name in names:
            name, n = f"{base}_{n}", n + 1
        names.append(name)
    return names


def _sql_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    return "text"


def _python(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def upload(path: Path) -> list[UploadResult]:
    """Store each sheet of the file as a Dataset, replacing any Dataset of the same name."""
    return [_store(name, path.name, frame) for name, frame in _read(path).items()]


def _store(name: str, origin: str, frame: pd.DataFrame) -> UploadResult:
    columns = _column_names(frame.columns)
    types = [_sql_type(frame[c]) for c in frame.columns]
    table = sql.Identifier(SCHEMA, name)
    with db.pool().connection() as conn, conn.transaction():
        replaced = bool(conn.execute("SELECT 1 FROM datasets WHERE name = %s", (name,)).fetchone())
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
        conn.execute(
            sql.SQL("CREATE TABLE {} ({})").format(
                table,
                sql.SQL(", ").join(
                    sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(t))
                    for c, t in zip(columns, types, strict=True)
                ),
            )
        )
        copy = sql.SQL("COPY {} ({}) FROM STDIN").format(
            table, sql.SQL(", ").join(map(sql.Identifier, columns))
        )
        with conn.cursor().copy(copy) as writer:
            for row in frame.itertuples(index=False, name=None):
                writer.write_row([_python(v) for v in row])
        conn.execute(
            """
            INSERT INTO datasets (name, origin, columns, row_count) VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET origin = excluded.origin,
                columns = excluded.columns, row_count = excluded.row_count, uploaded_at = now()
            """,
            (
                name,
                origin,
                Jsonb([{"name": c, "type": t} for c, t in zip(columns, types, strict=True)]),
                len(frame),
            ),
        )
    dataset = Dataset(name, origin, columns, types, len(frame))
    return UploadResult(dataset, replaced)


def list_datasets() -> list[Dataset]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT name, origin, columns, row_count FROM datasets ORDER BY name"
        ).fetchall()
    return [
        Dataset(
            r["name"],
            r["origin"],
            [c["name"] for c in r["columns"]],
            [c["type"] for c in r["columns"]],
            r["row_count"],
        )
        for r in rows
    ]


def run_query(
    query: str, *, row_limit: int = ROW_LIMIT, timeout_ms: int = STATEMENT_TIMEOUT_MS
) -> QueryResult:
    """Run one SQL statement as the read-only analyst role. Raises QueryError on failure."""
    try:
        with psycopg.connect(
            db.analyst_conninfo(timeout_ms), autocommit=True, connect_timeout=5
        ) as conn:
            cursor = conn.execute(query)
            if cursor.description is None:
                return QueryResult([], [], truncated=False)
            columns = [c.name for c in cursor.description]
            rows = cursor.fetchmany(row_limit + 1)
    except psycopg.errors.QueryCanceled as error:
        raise QueryError(
            f"The query took longer than {timeout_ms / 1000:g} seconds and was stopped."
        ) from error
    except psycopg.Error as error:
        message = error.diag.message_primary or str(error).strip()
        raise QueryError(f"The query failed: {message}") from error
    return QueryResult(columns, rows[:row_limit], truncated=len(rows) > row_limit)

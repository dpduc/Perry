# Perry Database Inspection & Query Guide

This guide explains how to inspect, query, and monitor Perry's PostgreSQL database using both **PowerShell (via Docker)** and **DBeaver (GUI)**.

---

## 1. Database Architecture & Connection Details

Perry uses a **PostgreSQL 17** database with the **`pgvector`** extension, running inside a Docker container defined in `compose.yaml`.

* **Docker Service / Container Name**: `perry-db-1`
* **Docker Image**: `pgvector/pgvector:pg17`
* **Host**: `127.0.0.1` (or `localhost`)
* **Port**: `5433` *(mapped from internal `5432` to avoid conflicting with any local Postgres)*
* **Database Name**: `perry`
* **Admin / App User**: `perry` / Password: `perry`
* **Read-Only Analyst Role**: `perry_analyst` / Password: `perry_analyst`
* **Full URL**: `postgresql://perry:perry@127.0.0.1:5433/perry`

### Where Data Lives on Your Hard Drive
* **Docker Volume**: `perry_pgdata`
  * Mounted in WSL 2 / Docker Desktop virtual disk at:
    `\\wsl$\docker-desktop-data\data\docker\volumes\perry_pgdata\_data`
* **App Data Cache**: Uploaded files and cached data outside Postgres live at:
  * `%APPDATA%\Perry` (Windows: `C:\Users\<username>\AppData\Roaming\Perry`)

---

## 2. Inspecting & Querying via PowerShell (Docker)

No database client installation is required. You can manage everything directly from PowerShell.

### Check if Database Container is Running
```powershell
docker ps --filter "name=perry-db-1"
```
If it is stopped, start it with:
```powershell
docker compose up -d db
```

---

### Option A: Interactive `psql` Shell

Open an interactive Postgres terminal:
```powershell
docker exec -it perry-db-1 psql -U perry -d perry
```

Once inside (`perry=#`), you can run queries or meta-commands:

| Meta-Command | Description |
| :--- | :--- |
| `\dt` | List all tables in the `public` schema |
| `\dt analytics.*` | List all uploaded tabular datasets |
| `\d <table_name>` | Describe columns and types of a specific table (e.g. `\d chunks`) |
| `\d+ <table_name>` | Detailed table description including indexes and sizes |
| `\dn` | List all schemas (`public`, `analytics`) |
| `\dx` | List installed extensions (verifies `vector`) |
| `\x` | Toggle expanded display (great for wide rows or long text) |
| `\q` | **Exit** back to PowerShell |

---

### Option B: Run One-Off Queries Directly

Execute queries directly from PowerShell without entering the interactive shell using `-c`:

```powershell
docker exec perry-db-1 psql -U perry -d perry -c "SELECT id, name, ingested_at FROM documents;"
```

---

## 3. Inspecting & Querying via DBeaver (GUI)

DBeaver provides a visual tree explorer, spreadsheet data editor, and SQL workspace.

### Navigating Database Objects
In the left **Database Navigator** panel:
1. Expand **`perry localhost:5433`** → **`Databases`** → **`perry`** → **`Schemas`**.
2. **`public` schema**: Contains Perry's primary application tables:
   * `documents`: Ingested files (PDFs, Markdown, text).
   * `chunks`: Text chunks, page numbers, and vector embeddings.
   * `threads`: Conversation Threads, their titles, and the User each belongs to (`user_id`).
   * `usage_ledger`: Token usage and cost per model call, with its Turn, Thread and User (`user_id`, empty outside a Turn).
   * `datasets`: Registry of uploaded CSV and Excel files.
   * `checkpoints*`: LangGraph conversation memory and graph checkpoints.
3. **`analytics` schema**: Contains real tables created dynamically when users upload CSV/spreadsheet datasets.

### Viewing & Filtering Table Data Visually
1. Double-click any table (e.g., `documents`, `usage_ledger`, or `chunks`).
2. Click the **Data** tab at the top.
3. Features available:
   * **Filter**: Click the filter bar at the top (e.g. `cost_usd > 0`).
   * **Sort**: Click any column header to sort ascending/descending.
   * **Export**: Right-click the grid → **Export Data** → Export to CSV, JSON, or Excel.

### Running SQL Queries
1. Press **`Ctrl + ]`** (or click the **SQL Editor** button in the toolbar).
2. Write or paste your SQL query.
3. Press **`Ctrl + Enter`** to execute the current statement (or **`Alt + X`** to run all statements).
4. View results in the bottom result grid.

---

## 4. Useful SQL Queries for Stats & Health Overview

Run these queries in either `psql` or DBeaver's SQL Editor:

### 1. Overall Database Size
```sql
SELECT 
    pg_size_pretty(pg_database_size('perry')) AS total_database_size;
```

---

### 2. Table Row Counts (Quick Overview)
```sql
SELECT 
    schemaname,
    relname AS table_name,
    n_live_tup AS estimated_rows
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

---

### 3. Disk Space Used per Table & Indexes
Shows how much storage each table and its indexes consume:
```sql
SELECT 
    relname AS table_name,
    pg_size_pretty(pg_table_size(relid)) AS data_size,
    pg_size_pretty(pg_indexes_size(relid)) AS index_size,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

---

### 4. Ingested Documents & Chunk Stats
Inspect which documents have been ingested and how many chunks each produced:
```sql
SELECT 
    d.id,
    d.name,
    d.ingested_at,
    COUNT(c.id) AS total_chunks,
    MIN(c.page) AS first_page,
    MAX(c.page) AS last_page
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
GROUP BY d.id, d.name, d.ingested_at
ORDER BY d.ingested_at DESC;
```

---

### 5. Inspect Chunk Content & Vector Embeddings
Preview chunk texts and verify embedding dimensions:
```sql
SELECT 
    c.id,
    d.name AS document_name,
    c.page,
    SUBSTRING(c.text FROM 1 FOR 100) AS text_preview,
    vector_dims(c.embedding) AS embedding_dimensions
FROM chunks c
JOIN documents d ON d.id = c.document_id
ORDER BY c.id ASC
LIMIT 10;
```

---

### 6. Conversation Threads Overview
List recent conversation threads:
```sql
SELECT 
    id,
    title,
    created_at
FROM threads
ORDER BY created_at DESC
LIMIT 20;
```

---

### 7. Token Usage & Cost Summary by Model
Aggregate total tokens and cost per model from `usage_ledger`:
```sql
SELECT 
    model,
    kind,
    COUNT(*) AS total_calls,
    SUM(input_tokens) AS total_input_tokens,
    SUM(output_tokens) AS total_output_tokens,
    ROUND(SUM(cost_usd), 4) AS total_cost_usd
FROM usage_ledger
GROUP BY model, kind
ORDER BY total_cost_usd DESC;
```

---

### 8. Recent Model Invocations
Inspect the latest LLM/embedding calls recorded:
```sql
SELECT 
    recorded_at,
    thread_id,
    kind,
    model,
    input_tokens,
    output_tokens,
    cost_usd
FROM usage_ledger
ORDER BY recorded_at DESC
LIMIT 15;
```

---

### 9. Uploaded Datasets (Analytics Schema)
List tabular datasets registered in Perry:
```sql
SELECT 
    name AS dataset_table,
    origin AS source_file,
    row_count,
    uploaded_at,
    columns
FROM datasets
ORDER BY uploaded_at DESC;
```
*(To query an uploaded dataset directly: `SELECT * FROM analytics.<dataset_name> LIMIT 50;`)*

---

## 5. Visualizing Database Stats

### Recommended Visual Metrics for Perry

| Domain | Metric / Insight | Best Chart Type | Reason |
| :--- | :--- | :--- | :--- |
| **Token & Cost Economics** | Cost distribution across Agents (`kind`: `knowledge`, `analysis`, `task`, `supervisor`) | **Donut / Pie Chart** | Instantly spots which agent is burning the most budget. |
| **Token & Cost Economics** | Input vs Output token ratio per model | **Stacked Bar Chart** | Identifies prompt verbosity vs model generation length. |
| **Token & Cost Economics** | Cumulative spend over time | **Line / Area Chart** | Monitors cost trends per hour or day. |
| **Knowledge Base** | Chunks per Document | **Horizontal Bar Chart** | Highlights document density and chunks ingested. |
| **Conversation Health** | Turns / Checkpoints per Thread | **Bar Chart** | Shows conversation depth (single questions vs deep multi-turn work). |
| **Storage & Infrastructure** | Disk space per table (data vs index vs vector) | **Bar / Treemap** | Tracks vector embedding (`vector_cosine_ops`) overhead. |

---

### Method A: Native Charts inside DBeaver (Zero Code)

DBeaver has a built-in charting engine right inside the query results window:

1. Open a new SQL Editor in DBeaver (`Ctrl + ]`).
2. Run any aggregation query below.
3. On the **right-hand vertical toolbar** of the result grid, click the **Chart** icon (or press **`Ctrl + Shift + F8`**).
4. Select the chart type (**Pie**, **Bar**, or **Line**) and set:
   * **Column / Category**: The text column (e.g. `kind` or `name`)
   * **Values**: The numeric column (e.g. `cost` or `chunks`)

#### Query 1: Agent Cost Breakdown (for Pie / Donut Chart)
```sql
SELECT kind, ROUND(SUM(cost_usd)::numeric, 4) AS cost_usd
FROM usage_ledger
GROUP BY kind
ORDER BY cost_usd DESC;
```

#### Query 2: Chunks per Document (for Bar Chart)
```sql
SELECT d.name, COUNT(c.id) AS chunks
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
GROUP BY d.name
ORDER BY chunks DESC;
```

#### Query 3: Table Disk Usage (for Bar Chart)
```sql
SELECT relname AS table_name, ROUND(pg_total_relation_size(relid) / 1024.0, 1) AS size_kb
FROM pg_stat_user_tables
ORDER BY size_kb DESC;
```

---

### Method B: Automated Interactive HTML Dashboard (One Command)

We have created a helper script [scripts/generate_db_report.py](file:///c:/Users/teflo/Desktop/WORK/Internship/Projects/Perry/scripts/generate_db_report.py) that queries the live database and generates an interactive, dark-themed dashboard using Chart.js:

```powershell
uv run python scripts/generate_db_report.py
```

This will automatically create `db_dashboard.html` and launch it in your default browser with:
* Summary KPI cards (Total DB size, LLM cost, total tokens, ingested docs & chunks).
* Interactive Donut chart for Agent cost distribution.
* Storage footprint bar chart per table.
* Chunks per document breakdown.
* Full table status overview.


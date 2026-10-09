"""Perry Database Visualizer: Generates a standalone, interactive HTML dashboard with Chart.js."""

import json
import os
import webbrowser
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://perry:perry@127.0.0.1:5433/perry")


def fetch_stats():
    stats = {}
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn, conn.cursor() as cur:
        # 1. Total DB Size
        cur.execute("SELECT pg_size_pretty(pg_database_size('perry')) AS total_size;")
        stats["db_size"] = cur.fetchone()["total_size"]

        # 2. Table row counts & sizes
        cur.execute("""
            SELECT 
                relname AS table_name,
                n_live_tup AS rows,
                pg_total_relation_size(relid) AS total_bytes,
                pg_size_pretty(pg_total_relation_size(relid)) AS pretty_size
            FROM pg_stat_user_tables
            ORDER BY total_bytes DESC;
        """)
        stats["tables"] = cur.fetchall()

        # 3. Documents and chunk counts
        cur.execute("""
            SELECT 
                d.name,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.name
            ORDER BY chunk_count DESC;
        """)
        stats["doc_chunks"] = cur.fetchall()

        # 4. Usage Ledger: Cost by Model
        cur.execute("""
            SELECT 
                model,
                ROUND(SUM(cost_usd)::numeric, 4) AS cost,
                SUM(input_tokens) AS input_tokens,
                SUM(output_tokens) AS output_tokens,
                COUNT(*) AS calls
            FROM usage_ledger
            GROUP BY model
            ORDER BY cost DESC;
        """)
        stats["usage_by_model"] = cur.fetchall()

        # 5. Usage Ledger: Cost by Agent Kind
        cur.execute("""
            SELECT 
                kind,
                ROUND(SUM(cost_usd)::numeric, 4) AS cost,
                COUNT(*) AS calls
            FROM usage_ledger
            GROUP BY kind
            ORDER BY cost DESC;
        """)
        stats["usage_by_kind"] = cur.fetchall()

        # 6. Usage timeline (by day or hour)
        cur.execute("""
            SELECT 
                to_char(recorded_at, 'YYYY-MM-DD HH24:00') AS time_bucket,
                ROUND(SUM(cost_usd)::numeric, 5) AS cost,
                SUM(input_tokens + output_tokens) AS total_tokens
            FROM usage_ledger
            GROUP BY time_bucket
            ORDER BY time_bucket ASC
            LIMIT 30;
        """)
        stats["usage_timeline"] = cur.fetchall()

        # 7. Checkpoints per thread
        cur.execute("""
            SELECT 
                t.title,
                t.created_at,
                t.id
            FROM threads t
            ORDER BY t.created_at DESC
            LIMIT 10;
        """)
        stats["recent_threads"] = cur.fetchall()

    return stats


def generate_html(stats, output_path: Path):
    # Prepare chart datasets
    table_labels = [r["table_name"] for r in stats["tables"]]
    table_rows = [r["rows"] for r in stats["tables"]]
    table_sizes_kb = [round(r["total_bytes"] / 1024, 1) for r in stats["tables"]]

    doc_labels = [r["name"] for r in stats["doc_chunks"]] or ["No documents"]
    doc_chunk_counts = [r["chunk_count"] for r in stats["doc_chunks"]] or [0]

    model_costs = [float(r["cost"]) for r in stats["usage_by_model"]] or [0.0]
    model_tokens = [r["input_tokens"] + r["output_tokens"] for r in stats["usage_by_model"]] or [0]

    kind_labels = [r["kind"] for r in stats["usage_by_kind"]] or ["None"]
    kind_costs = [float(r["cost"]) for r in stats["usage_by_kind"]] or [0.0]


    total_cost = sum(model_costs)
    total_tokens = sum(model_tokens)
    total_docs = len(stats["doc_chunks"])
    total_chunks = sum(doc_chunk_counts)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Perry Database Analytics & Health Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --border: #334155;
      --accent: #38bdf8;
      --accent-purple: #a855f7;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}
    .header h1 {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
    .header .meta {{ color: var(--text-muted); font-size: 14px; }}
    .grid-kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }}
    .kpi-title {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-value {{ font-size: 26px; font-weight: 700; margin-top: 6px; color: #fff; }}
    .grid-charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .chart-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }}
    .chart-card h2 {{
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .chart-container {{ position: relative; height: 260px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 8px;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{ color: var(--text-muted); font-weight: 600; }}
    tr:hover {{ background: rgba(255, 255, 255, 0.03); }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 9999px;
      font-size: 12px;
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Perry Database Analytics & Health</h1>
      <div class="meta">PostgreSQL 17 + pgvector (Container: <code>perry-db-1</code> on port 5433)</div>
    </div>
    <div class="badge">Live Snapshot</div>
  </div>

  <div class="grid-kpi">
    <div class="kpi-card">
      <div class="kpi-title">Total Database Size</div>
      <div class="kpi-value">{stats["db_size"]}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Total LLM Cost</div>
      <div class="kpi-value" style="color: var(--accent-emerald);">${total_cost:.4f}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Total Tokens Logged</div>
      <div class="kpi-value" style="color: var(--accent);">{total_tokens:,}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Ingested Documents</div>
      <div class="kpi-value">{total_docs} <span style="font-size: 14px; color: var(--text-muted);">({total_chunks} chunks)</span></div>
    </div>
  </div>

  <div class="grid-charts">
    <!-- Cost by Agent Kind -->
    <div class="chart-card">
      <h2>Agent Cost Distribution (Kind)</h2>
      <div class="chart-container">
        <canvas id="kindChart"></canvas>
      </div>
    </div>

    <!-- Storage by Table -->
    <div class="chart-card">
      <h2>Storage Footprint per Table (KB)</h2>
      <div class="chart-container">
        <canvas id="storageChart"></canvas>
      </div>
    </div>

    <!-- Chunks per Document -->
    <div class="chart-card">
      <h2>Chunks per Document</h2>
      <div class="chart-container">
        <canvas id="docChart"></canvas>
      </div>
    </div>

    <!-- Table Row Counts -->
    <div class="chart-card">
      <h2>Row Count per Table</h2>
      <div class="chart-container">
        <canvas id="rowsChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Tables Section -->
  <div class="chart-card">
    <h2>Detailed Table Statistics</h2>
    <table>
      <thead>
        <tr>
          <th>Table Name</th>
          <th>Estimated Rows</th>
          <th>Total Size on Disk</th>
        </tr>
      </thead>
      <tbody>
        {"".join(f"<tr><td><code>{t['table_name']}</code></td><td>{t['rows']:,}</td><td>{t['pretty_size']}</td></tr>" for t in stats["tables"])}
      </tbody>
    </table>
  </div>

  <script>
    const colors = ['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ec4899', '#6366f1', '#14b8a6', '#f43f5e'];

    // 1. Kind Chart
    new Chart(document.getElementById('kindChart'), {{
      type: 'doughnut',
      data: {{
        labels: {json.dumps(kind_labels)},
        datasets: [{{
          data: {json.dumps(kind_costs)},
          backgroundColor: colors,
          borderWidth: 0
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }}
      }}
    }});

    // 2. Storage Chart
    new Chart(document.getElementById('storageChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(table_labels)},
        datasets: [{{
          label: 'Total Size (KB)',
          data: {json.dumps(table_sizes_kb)},
          backgroundColor: '#38bdf8',
          borderRadius: 6
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
        }}
      }}
    }});

    // 3. Document Chunks Chart
    new Chart(document.getElementById('docChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(doc_labels)},
        datasets: [{{
          label: 'Chunks',
          data: {json.dumps(doc_chunk_counts)},
          backgroundColor: '#10b981',
          borderRadius: 6
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }}
        }}
      }}
    }});

    // 4. Rows Chart
    new Chart(document.getElementById('rowsChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(table_labels)},
        datasets: [{{
          label: 'Rows',
          data: {json.dumps(table_rows)},
          backgroundColor: '#a855f7',
          borderRadius: 6
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Report generated at: {output_path.resolve()}")


def main():
    report_file = Path("db_dashboard.html")
    stats = fetch_stats()
    generate_html(stats, report_file)
    print("Opening dashboard in browser...")
    webbrowser.open(report_file.resolve().as_uri())


if __name__ == "__main__":
    main()

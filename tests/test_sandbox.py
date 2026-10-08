"""The Sandbox against real Docker: skipped when Docker isn't running."""

from perry import sandbox


def test_the_output_of_the_code_comes_back(sandbox_image):
    result = sandbox.run("print(6 * 7)")

    assert result.ok
    assert result.output.strip() == "42"


def test_a_script_past_the_time_limit_is_killed_and_reported(sandbox_image):
    result = sandbox.run("while True:\n    pass", time_limit_s=3)

    assert not result.ok
    assert "longer than 3 seconds" in result.error
    assert sandbox.running_containers() == []


def test_the_network_is_unreachable_from_inside(sandbox_image):
    code = """
import socket
for host, port in [("1.1.1.1", 53), ("pypi.org", 443), ("host.docker.internal", 5433)]:
    try:
        socket.create_connection((host, port), timeout=3).close()
        print("reached", host)
    except OSError as error:
        print("blocked", host, type(error).__name__)
"""
    result = sandbox.run(code)

    assert result.ok, result.error
    assert "reached" not in result.output
    assert result.output.count("blocked") == 3


def test_the_host_env_credentials_and_project_files_are_invisible(sandbox_image, monkeypatch):
    monkeypatch.setenv("PERRY_SECRET_PROBE", "host-secret-value")
    code = """
import os
print("ENV", sorted(os.environ))
print("SECRET", "host-secret-value" in str(os.environ))
print("WORK", sorted(os.listdir(".")))
hits = []
for root, dirs, files in os.walk("/"):
    dirs[:] = [d for d in dirs if os.path.join(root, d) not in ("/proc", "/sys", "/dev")]
    hits += [os.path.join(root, f) for f in files
             if f in (".env", "GLOSSARY.md", "supervisor.py", "google_token.json")]
print("HITS", hits)
"""
    result = sandbox.run(code, {"sales.csv": b"region,amount\nNorth,1\n"})

    assert result.ok, result.error
    assert "SECRET False" in result.output
    assert "GEMINI_API_KEY" not in result.output
    assert "DATABASE_URL" not in result.output
    assert "WORK ['sales.csv']" in result.output
    assert "HITS []" in result.output


def test_input_files_are_readable_and_saved_charts_and_tables_come_back(sandbox_image):
    code = """
import pandas as pd
import matplotlib.pyplot as plt
sales = pd.read_csv("sales.csv")
totals = sales.groupby("region")["amount"].sum()
print(totals.to_string())
totals.to_csv("totals.csv")
totals.plot.bar()
plt.savefig("totals.png")
"""
    data = b"region,amount\nNorth,100\nSouth,250\nNorth,50\n"
    result = sandbox.run(code, {"sales.csv": data})

    assert result.ok, result.error
    assert "North    150" in result.output
    assert "South    250" in result.output
    assert len(result.charts) == 1
    assert result.charts[0].startswith(b"\x89PNG")
    assert "North,150" in result.tables["totals.csv"]


def test_a_failing_script_reports_its_error(sandbox_image):
    result = sandbox.run("print('before')\n1 / 0")

    assert not result.ok
    assert "ZeroDivisionError" in result.error
    assert "before" in result.output

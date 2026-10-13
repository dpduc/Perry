"""Serving Perry over ngrok: the HTTPS proxy in front of it, and the start script's checks."""

import json

import gradio as gr
import pytest
from fastapi.testclient import TestClient
from logins import OWNER_PASSWORD, OWNER_USERNAME, log_in

DOMAIN = "perry-example.ngrok-free.app"
# What ngrok's agent adds to each request it forwards from https://DOMAIN to 127.0.0.1.
THROUGH_NGROK = {"host": DOMAIN, "x-forwarded-host": DOMAIN, "x-forwarded-proto": "https"}


def file_url_in_answer(client, headers: dict) -> str:
    started = client.post("/chat/gradio_api/call/a_file", json={"data": []}, headers=headers)
    event_id = started.json()["event_id"]
    stream = client.get(f"/chat/gradio_api/call/a_file/{event_id}", headers=headers).text
    data = stream.split("event: complete\ndata: ", 1)[1].splitlines()[0]
    return json.loads(data)[0]["url"]


@pytest.mark.parametrize(
    "headers, root",
    [({}, "http://testserver/chat"), (THROUGH_NGROK, f"https://{DOMAIN}/chat")],
    ids=["locally", "through ngrok"],
)
def test_files_in_answers_link_under_the_chat_path(owner, tmp_path, headers, root):
    """A Chart in an answer is a file Gradio links to: the link must reach the mounted UI."""
    from perry import web

    chart = tmp_path / "chart.txt"
    chart.write_text("a chart")
    with gr.Blocks() as blocks:
        shown = gr.File()
        gr.Button().click(lambda: str(chart), None, shown, api_name="a_file")
    with TestClient(web.create_app(blocks)) as client:
        log_in(client, OWNER_USERNAME, OWNER_PASSWORD)

        url = file_url_in_answer(client, headers)

    assert url.startswith(f"{root}/gradio_api/file=")


def test_through_ngrok_the_chat_page_calls_back_to_the_public_https_url(owner_client):
    config = owner_client.get("/chat/config", headers=THROUGH_NGROK).json()

    assert config["root"] == f"https://{DOMAIN}/chat"


def served_by_main(monkeypatch, argv: list[str]):
    """What `python -m perry.web <argv>` would serve, without starting the server."""
    import uvicorn

    from perry import web

    served = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: served.update(app=app))
    web.main(argv)
    return served["app"]


@pytest.mark.parametrize("argv, secure", [([], False), (["--https"], True)])
def test_the_session_cookie_is_secure_when_served_over_https(owner, monkeypatch, argv, secure):
    with TestClient(served_by_main(monkeypatch, argv)) as browser:
        response = log_in(browser, OWNER_USERNAME, OWNER_PASSWORD)

    cookie = response.headers["set-cookie"]
    assert cookie.startswith("perry_session=")
    assert ("secure" in cookie.lower()) is secure


@pytest.fixture
def ollama_at(monkeypatch):
    """Points Perry's local models at a URL."""
    from perry import models

    def point(url: str) -> str:
        monkeypatch.setattr(models, "OLLAMA_URL", url)
        return url

    return point


@pytest.fixture
def listening_url():
    """A local server that answers every GET, as a running Ollama would."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Answer(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"models": []}')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def a_closed_port_url() -> str:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_unreachable_local_models_are_named_with_their_url(ollama_at):
    from perry import models

    url = ollama_at(a_closed_port_url())

    problem = models.local_models_problem()

    assert problem is not None
    assert url in problem
    assert "Ollama" in problem


def test_reachable_local_models_are_no_problem(ollama_at, listening_url):
    from perry import models

    ollama_at(listening_url)

    assert models.local_models_problem() is None


# The start script (scripts/serve.py), with the programs it runs faked.


class FakeProcess:
    def __init__(self, argv):
        self.argv = argv
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.terminated = True


@pytest.fixture
def start_script(monkeypatch, ollama_at, listening_url):
    """scripts/serve.py with everything in place: the settings, ngrok installed, Ollama
    and Postgres up. It records what it runs instead of running it. Tests take one piece
    away to see it refused."""
    import shutil
    import subprocess

    from scripts import serve

    monkeypatch.setenv("PERRY_NGROK_DOMAIN", DOMAIN)
    ollama_at(listening_url)
    monkeypatch.setattr(shutil, "which", lambda name: f"C:/bin/{name}.exe")
    started = []

    def popen(argv, **options):
        started.append(FakeProcess(argv))
        return started[-1]

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", lambda argv, **options: started.append(argv))
    serve.started = started
    return serve


def test_without_an_ngrok_domain_it_refuses_and_starts_nothing(start_script, monkeypatch):
    monkeypatch.delenv("PERRY_NGROK_DOMAIN")

    with pytest.raises(SystemExit, match="PERRY_NGROK_DOMAIN"):
        start_script.main()

    assert start_script.started == []


def test_without_a_session_secret_it_refuses_and_starts_nothing(start_script, monkeypatch):
    monkeypatch.delenv("PERRY_SESSION_SECRET")

    with pytest.raises(SystemExit, match="PERRY_SESSION_SECRET"):
        start_script.main()

    assert start_script.started == []


def test_without_ngrok_installed_it_refuses(start_script, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(SystemExit, match="ngrok"):
        start_script.main()

    assert start_script.started == []


def test_without_an_owner_it_refuses_and_launches_nothing(start_script, database):
    from perry import users

    with pytest.raises(SystemExit, match="create-owner"):
        start_script.main()

    assert users.owner() is None
    assert start_script.started == []


def test_without_ollama_it_refuses_and_launches_nothing(start_script, owner, ollama_at):
    ollama_at(a_closed_port_url())

    with pytest.raises(SystemExit, match="Ollama"):
        start_script.main()

    assert start_script.started == []


def test_it_launches_perry_locally_then_the_tunnel_and_prints_the_public_url(
    start_script, owner, monkeypatch, listening_url, capsys
):
    import subprocess

    # Perry answers on its port once it has been launched.
    monkeypatch.setattr(start_script, "PERRY_URL", a_closed_port_url())
    fake_popen = subprocess.Popen

    def popen(argv, **options):
        monkeypatch.setattr(start_script, "PERRY_URL", listening_url)
        return fake_popen(argv, **options)

    monkeypatch.setattr(subprocess, "Popen", popen)

    start_script.main()

    perry, ngrok = start_script.started
    assert perry.argv[1:] == ["-m", "perry.web", "--https"]
    assert ngrok.argv[:3] == ["ngrok", "http", f"--url={DOMAIN}"]
    assert ngrok.argv[-1] == "127.0.0.1:7860"
    assert f"https://{DOMAIN}/" in capsys.readouterr().out
    assert perry.terminated  # ngrok stopped, so Perry is stopped too


def test_the_domain_may_be_given_as_a_url(start_script, monkeypatch):
    monkeypatch.setenv("PERRY_NGROK_DOMAIN", f"https://{DOMAIN}/")

    assert start_script.ngrok_domain() == DOMAIN

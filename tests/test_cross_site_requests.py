"""Cross-site request forgery: a page on another site can't make a logged-in browser
change anything in Perry. A state-changing request whose Origin (or, without one, its
Referer) names another site is refused.

A request with neither header is let through: browsers send Origin on every cross-site
POST, so only non-browser clients (which can't carry a victim's cookie) send neither."""

import json

import gradio as gr
from fastapi.testclient import TestClient
from logins import OWNER_PASSWORD, OWNER_USERNAME, invites_listed, is_logged_in, log_in

ELSEWHERE = "https://evil.example"
DOMAIN = "perry-example.ngrok-free.app"
THROUGH_NGROK = {"host": DOMAIN, "x-forwarded-host": DOMAIN, "x-forwarded-proto": "https"}


def create_invite(client, **headers):
    return client.post("/owner/invites", headers=headers, follow_redirects=False)


def test_a_post_from_another_site_is_refused(owner_client):
    response = create_invite(owner_client, origin=ELSEWHERE)

    assert response.status_code == 403
    assert invites_listed(owner_client) == []


def test_a_post_referred_by_another_site_is_refused(owner_client):
    response = create_invite(owner_client, referer=f"{ELSEWHERE}/page")

    assert response.status_code == 403
    assert invites_listed(owner_client) == []


def test_a_post_from_an_opaque_origin_is_refused(owner_client):
    """e.g. a sandboxed iframe, which sends `Origin: null`."""
    assert create_invite(owner_client, origin="null").status_code == 403


def test_logging_in_from_another_site_is_refused(client):
    response = client.post(
        "/login",
        data={"username": OWNER_USERNAME, "password": OWNER_PASSWORD},
        headers={"origin": ELSEWHERE},
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert not is_logged_in(client)


def test_logging_out_from_another_site_is_refused(owner_client):
    owner_client.post("/logout", headers={"origin": ELSEWHERE})

    assert is_logged_in(owner_client)


def test_a_post_from_perrys_own_pages_works(owner_client):
    assert create_invite(owner_client, origin="http://testserver").status_code == 303
    assert create_invite(owner_client, referer="http://testserver/owner").status_code == 303
    assert len(invites_listed(owner_client)) == 2


def test_a_post_with_neither_origin_nor_referer_works(owner_client):
    assert create_invite(owner_client).status_code == 303
    assert len(invites_listed(owner_client)) == 1


def test_a_get_from_another_site_still_works(owner_client):
    """Links to Perry from elsewhere still open it; GETs change nothing."""
    assert owner_client.get("/owner", headers={"origin": ELSEWHERE}).status_code == 200


def call_gradio(client, headers: dict) -> int:
    """Call a Gradio event over its HTTP API, as the chat page does; the status code."""
    started = client.post("/chat/gradio_api/call/hello", json={"data": []}, headers=headers)
    if started.status_code != 200:
        return started.status_code
    event_id = started.json()["event_id"]
    stream = client.get(f"/chat/gradio_api/call/hello/{event_id}", headers=headers).text
    assert json.loads(stream.split("event: complete\ndata: ", 1)[1].splitlines()[0]) == ["hi"]
    return 200


def gradio_client() -> TestClient:
    from perry import web

    with gr.Blocks() as blocks:
        shown = gr.Textbox()
        gr.Button().click(lambda: "hi", None, shown, api_name="hello")
    return TestClient(web.create_app(blocks))


def test_the_chat_page_calls_gradio_from_its_own_origin(owner):
    with gradio_client() as client:
        log_in(client, OWNER_USERNAME, OWNER_PASSWORD)

        assert call_gradio(client, {"origin": "http://testserver"}) == 200
        assert call_gradio(client, {"origin": ELSEWHERE}) == 403


def test_the_chat_page_calls_gradio_through_ngrok(owner):
    """ngrok forwards the public Host, which the page's Origin then matches."""
    with gradio_client() as client:
        log_in(client, OWNER_USERNAME, OWNER_PASSWORD)

        assert call_gradio(client, THROUGH_NGROK | {"origin": f"https://{DOMAIN}"}) == 200

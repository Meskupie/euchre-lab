import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["EUCHRE_PROGRESS"] = str(tmp_path_factory.mktemp("data") / "progress.json")
    app = importlib.import_module("server.app")
    return TestClient(app.app)


def test_state_and_training_flow(client):
    deal = client.get("/api/random_deal", params={"seed": 1, "dealer": 0}).json()
    r = client.post("/api/state", json={"deal": deal, "actions": []})
    assert r.status_code == 200
    body = r.json()
    ev = body["eval"]
    assert body["state"]["phase"] == "bid1" and ev["seat"] == 1
    assert {a["action"]["type"] for a in ev["actions"]} == {"pass", "order"}
    sit = ev["situation"]
    assert sit["context"].startswith("Round 1")

    situation = {k: sit[k] for k in ("key", "title", "context", "answer")}
    rec = client.post("/api/progress/record", json={"situation": situation, "loss": 0.0}).json()
    assert rec["is_correct"] and rec["streak"] == 1
    assert len(client.get("/api/progress").json()["situations"]) == 1


def test_play_a_whole_hand_with_ai_moves(client):
    deal = client.get("/api/random_deal", params={"seed": 2}).json()
    actions = []
    for _ in range(40):
        body = client.post("/api/state", json={"deal": deal, "actions": actions}).json()
        if body["state"]["done"]:
            break
        actions.append(body["eval"]["ai_action"])
    assert body["state"]["done"]
    assert len(body["history"]) == len(actions)


def test_illegal_action_rejected(client):
    deal = client.get("/api/random_deal", params={"seed": 3}).json()
    r = client.post("/api/state", json={"deal": deal, "actions": [{"type": "card", "card": deal["hands"][0][0]}]})
    assert r.status_code == 400


def test_deep_analysis(client):
    deal = client.get("/api/random_deal", params={"seed": 4}).json()
    r = client.post("/api/analyze", json={"deal": deal, "actions": [], "samples": 16})
    assert r.status_code == 200 and len(r.json()["actions"]) == 3

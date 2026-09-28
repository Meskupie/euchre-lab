"""Per-situation training progress, persisted to a JSON file next to the app.

A situation is *mastered* once it has been answered correctly `streak_to_master` times in a
row; mastered situations are auto-played, except that a fraction `review_rate` of them are
still shown so forgotten lessons resurface. Any mistake resets the streak.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

DEFAULT_SETTINGS = {"streak_to_master": 3, "review_rate": 0.1, "tolerance": 0.03}


class Progress:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"settings": dict(DEFAULT_SETTINGS), "situations": {}}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                self.data["situations"] = loaded.get("situations", {})
                self.data["settings"].update(loaded.get("settings", {}))
            except (OSError, json.JSONDecodeError):
                pass

    @property
    def settings(self) -> dict:
        return self.data["settings"]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)

    def mastered(self, key: str) -> bool:
        s = self.data["situations"].get(key)
        return bool(s) and s["streak"] >= self.settings["streak_to_master"]

    def should_skip(self, key: str) -> bool:
        return self.mastered(key) and random.random() >= self.settings["review_rate"]

    def record(self, sit: dict, loss: float) -> dict:
        correct = loss <= self.settings["tolerance"]
        with self.lock:
            s = self.data["situations"].setdefault(sit["key"], {
                "title": sit["title"], "context": sit["context"], "answer": sit["answer"],
                "seen": 0, "correct": 0, "streak": 0, "lost": 0.0,
            })
            s["seen"] += 1
            s["correct"] += correct
            s["streak"] = s["streak"] + 1 if correct else 0
            s["lost"] = round(s["lost"] + max(loss, 0.0), 4)
            s["last"] = time.time()
            self._save()
            return {**s, "key": sit["key"], "is_correct": correct, "mastered": self.mastered(sit["key"])}

    def summary(self) -> dict:
        items = [{"key": k, **v, "mastered": self.mastered(k)} for k, v in self.data["situations"].items()]
        return {"settings": self.settings, "situations": items}

    def update_settings(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                if k in DEFAULT_SETTINGS and v is not None:
                    self.data["settings"][k] = type(DEFAULT_SETTINGS[k])(v)
            self._save()

    def reset(self) -> None:
        with self.lock:
            self.data["situations"] = {}
            self._save()

"""
VibeThinker Studio - FastAPI server.

Serves a single-page UI and streams model output over Server-Sent Events for
both inference modes:

    POST /api/generate       -> non-CLR single pass (token stream)
    POST /api/generate_clr   -> CLR multi-sample + claim verification (event stream)
    GET  /api/status         -> model / device readiness
    POST /api/load           -> trigger model load (so the UI can show progress)

Run:  python app.py   (then open http://127.0.0.1:8000)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

# Use the exported Windows CA bundle if present (fixes corporate-proxy TLS).
_ca = Path(__file__).parent / "win-ca-bundle.pem"
if _ca.exists():
    os.environ.setdefault("SSL_CERT_FILE", str(_ca))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_ca))

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from engine import ENGINE, GenParams, extract_answer

app = FastAPI(title="VibeThinker Studio")
WEB = Path(__file__).parent / "web"


class GenRequest(BaseModel):
    prompt: str
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 8192
    n_samples: int = 4  # CLR only


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# Only one generation runs at a time (single GPU). A new request — or an explicit
# /api/stop — supersedes the previous run by setting its stop event, so a reloaded
# page or the Stop button frees the GPU instead of blocking the queue. (Starlette
# can't throw into a sync generator running in its threadpool, so disconnect alone
# is not enough.)
_CUR = {"stop": None}
_CUR_LOCK = threading.Lock()


def new_stop() -> threading.Event:
    with _CUR_LOCK:
        if _CUR["stop"] is not None:
            _CUR["stop"].set()
        ev = threading.Event()
        _CUR["stop"] = ev
        return ev


@app.post("/api/stop")
def stop():
    with _CUR_LOCK:
        if _CUR["stop"] is not None:
            _CUR["stop"].set()
    return {"stopped": True}


@app.get("/api/status")
def status():
    return {
        "ready": ENGINE.ready,
        "model": "VibeThinker-3B",
        "device": ENGINE.device_info(),
        "model_present": (Path(ENGINE.model_dir) / "config.json").exists(),
        "load_error": ENGINE.load_error,
    }


@app.post("/api/load")
def load():
    try:
        ENGINE.load()
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": repr(e)}


@app.post("/api/generate")
def generate(req: GenRequest):
    params = GenParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.temperature > 0,
    )

    stop = new_stop()

    def stream():
        yield sse({"type": "phase", "phase": "thinking"})
        parts = []
        try:
            for chunk in ENGINE.stream(req.prompt, params, stop_event=stop):
                parts.append(chunk)
                yield sse({"type": "token", "text": chunk})
        except GeneratorExit:  # client disconnected -> stop the GPU work
            stop.set()
            raise
        except Exception as e:
            yield sse({"type": "error", "message": repr(e)})
            return
        full = "".join(parts)
        yield sse({"type": "done", "answer": extract_answer(full),
                   "tokens": ENGINE.count_tokens(full)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/generate_clr")
def generate_clr(req: GenRequest):
    params = GenParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_new_tokens=req.max_new_tokens,
    )
    n = max(2, min(req.n_samples, 8))

    stop = new_stop()

    def stream():
        try:
            for event in ENGINE.clr_stream(req.prompt, params, n_samples=n,
                                           stop_event=stop):
                yield sse(event)
        except GeneratorExit:  # client disconnected -> stop the GPU work
            stop.set()
            raise
        except Exception as e:
            yield sse({"type": "error", "message": repr(e)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

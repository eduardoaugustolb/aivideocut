# pyright: basic
import asyncio
import os
import platform
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

load_dotenv()

from aivideocut.configs import OUTPUT_DIR_PATH  # noqa: E402
from aivideocut.gem_article import gem_create_article  # noqa: E402
from aivideocut.gem_srt import fix_srt_typos  # noqa: E402
from aivideocut.gem_srt_english import gem_translate_srt_to_en  # noqa: E402
from aivideocut.gem_summary import generate_summary  # noqa: E402
from aivideocut.gem_yt_chapters import gem_yt_chapters  # noqa: E402
from aivideocut.gem_yt_seo import gem_yt_seo  # noqa: E402
from aivideocut.sil2 import run_single_file  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="aivideocut")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_run_lock = threading.Lock()
_runs: dict[str, "Run"] = {}

SETTINGS_KEYS = ("WHISPER_API_URL", "WHISPER_API_TOKEN", "GEMINI_API_KEY", "NVIDIA_API_KEY")

GENERATE_STEPS: dict[str, Callable[[], Any]] = {
    "fix_srt": fix_srt_typos,
    "translate_en": gem_translate_srt_to_en,
    "summary": generate_summary,
    "article": gem_create_article,
    "chapters": gem_yt_chapters,
    "seo": gem_yt_seo,
}


# ==========================
# Background run machinery
# ==========================


class Run:
    """Runs `target` in a background thread with a persistent log buffer.

    The run is fully decoupled from any WebSocket connection: it keeps going
    even if every viewer disconnects (e.g. the laptop went to sleep). A
    (re)connecting client just reads `log_text` from the start — there's no
    per-connection cursor to lose track of.
    """

    def __init__(self, run_id: str, target: Callable[[], Any]) -> None:
        self.id = run_id
        self.log_text = ""
        self._log_lock = threading.Lock()
        self.status: Literal["running", "done", "error"] = "running"
        self.error: str | None = None
        self.result: Any = None
        self.done_event = threading.Event()
        self._target = target

    def _append_log(self, text: str) -> None:
        with self._log_lock:
            self.log_text += text

    def snapshot(self) -> str:
        with self._log_lock:
            return self.log_text

    def _pump_fd(self, read_fd: int) -> None:
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            self._append_log(chunk.decode("utf-8", errors="replace"))
        os.close(read_fd)

    def run(self) -> None:
        # Redirect at the OS file-descriptor level (not just sys.stdout) so
        # subprocess output (ffmpeg, auto-editor) — which inherits the real
        # fd, not Python's sys.stdout object — also reaches the log stream.
        sys.stdout.flush()
        sys.stderr.flush()

        read_fd, write_fd = os.pipe()
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)

        # Line-buffer Python's own prints during the run so they interleave
        # in the right order with subprocess output instead of sitting in a
        # block buffer until the run finishes.
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

        pump_thread = threading.Thread(target=self._pump_fd, args=(read_fd,), daemon=True)
        pump_thread.start()

        try:
            self.result = self._target()
            self.status = "done"
        except Exception as exc:  # noqa: BLE001
            self.status = "error"
            self.error = str(exc)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            pump_thread.join(timeout=5)
            self.done_event.set()


_current_run_id: str | None = None


def _start_run(target: Callable[[], Any]) -> str:
    global _current_run_id  # noqa: PLW0603

    if not _run_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="Já existe uma execução em andamento"
        )

    run_id = str(uuid4())
    run = Run(run_id, target)
    _runs[run_id] = run
    _current_run_id = run_id

    def _runner() -> None:
        try:
            run.run()
        finally:
            _run_lock.release()

    threading.Thread(target=_runner, daemon=True).start()
    return run_id


# ==========================
# Static / index
# ==========================


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


# ==========================
# Settings (Colab API url/token, Gemini/NVIDIA keys)
# ==========================


def _read_env_file() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        values[key.strip()] = raw_value.strip().strip("'\"")

    return values


def _write_env_file(values: dict[str, str]) -> None:
    lines = []
    seen = set()

    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                lines.append(line)
                continue

            key = stripped.split("=", 1)[0].strip()
            if key in values:
                lines.append(f"{key}='{values[key]}'")
                seen.add(key)
            else:
                lines.append(line)

    for key, value in values.items():
        if key not in seen:
            lines.append(f"{key}='{value}'")

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    values = _read_env_file()
    return {
        "WHISPER_API_URL": values.get("WHISPER_API_URL", ""),
        "WHISPER_API_TOKEN_set": bool(values.get("WHISPER_API_TOKEN")),
        "GEMINI_API_KEY_set": bool(values.get("GEMINI_API_KEY")),
        "NVIDIA_API_KEY_set": bool(values.get("NVIDIA_API_KEY")),
    }


class SettingsRequest(BaseModel):
    values: dict[str, str]


@app.post("/api/settings")
def save_settings(req: SettingsRequest) -> dict[str, bool]:
    clean = {
        key: value.strip()
        for key, value in req.values.items()
        if key in SETTINGS_KEYS and value.strip()
    }
    if not clean:
        return {"ok": True}

    _write_env_file(clean)
    for key, value in clean.items():
        os.environ[key] = value

    return {"ok": True}


# ==========================
# Native file dialog
# ==========================


@app.post("/api/browse")
def browse() -> dict[str, str | None]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="tkinter não disponível nesse sistema — digite o caminho manualmente",
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Selecione o vídeo",
        filetypes=[("Vídeos", "*.mp4 *.mov *.mkv"), ("Todos os arquivos", "*.*")],
    )
    root.destroy()

    return {"path": path or None}


def _open_native(path: Path) -> None:
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", str(path)])
    elif system == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)])


class OpenRequest(BaseModel):
    path: str


@app.post("/api/open")
def open_path(req: OpenRequest) -> dict[str, bool]:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Caminho não encontrado")
    _open_native(path)
    return {"ok": True}


# ==========================
# Pipeline (video cut + transcribe)
# ==========================


class PipelineRequest(BaseModel):
    input_path: str
    fix_codecs: bool = True
    normalize_audio: bool = True
    cut_audio_silences: bool = True
    cut_speech_silences: bool = True
    transcribe_audio: bool = True
    enhance_before_transcribe: bool = False
    force: bool = False


@app.post("/api/pipeline/start")
def start_pipeline(req: PipelineRequest) -> dict[str, str]:
    input_path = Path(req.input_path).expanduser()
    if not input_path.is_file():
        raise HTTPException(status_code=400, detail="Arquivo de vídeo não encontrado")

    def _target() -> list[str]:
        result = run_single_file(
            input_path=input_path,
            fix_codecs=req.fix_codecs,
            normalize_audio=req.normalize_audio,
            cut_audio_silences=req.cut_audio_silences,
            cut_speech_silences=req.cut_speech_silences,
            transcribe_audio=req.transcribe_audio,
            enhance_before_transcribe=req.enhance_before_transcribe,
            force=req.force,
        )
        return [str(p) for p in result]

    run_id = _start_run(_target)
    return {"run_id": run_id}


# ==========================
# Content generation (Gemini/NVIDIA steps over the transcription)
# ==========================


@app.post("/api/generate/{step}")
def start_generate(step: str) -> dict[str, str]:
    target = GENERATE_STEPS.get(step)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Etapa desconhecida: {step}")

    run_id = _start_run(target)
    return {"run_id": run_id}


# ==========================
# Run status via WebSocket
# ==========================


@app.websocket("/api/runs/{run_id}/ws")
async def run_ws(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    run = _runs.get(run_id)
    if run is None:
        await websocket.send_json({"type": "error", "message": "Execução não encontrada"})
        await websocket.close()
        return

    # Polling on a growing buffer (instead of draining a single-consumer
    # queue) means a client that reconnects after a dropped connection
    # (laptop sleep, wifi blip...) just gets replayed the full log from the
    # start — the background run itself never depended on anyone listening.
    last_sent: str | None = None
    try:
        while True:
            current = run.snapshot()
            if current != last_sent:
                await websocket.send_json({"type": "log", "text": current, "replace": True})
                last_sent = current

            if run.done_event.is_set():
                break

            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        return

    await websocket.send_json(
        {
            "type": "done",
            "status": run.status,
            "error": run.error,
            "result": run.result,
        }
    )


@app.get("/api/runs/current")
def get_current_run() -> dict[str, Any]:
    if _current_run_id is None or _current_run_id not in _runs:
        return {"run_id": None}

    run = _runs[_current_run_id]
    return {"run_id": run.id, "status": run.status, "error": run.error}


# ==========================
# Generated files listing / preview
# ==========================


@app.get("/api/files")
def list_files() -> list[dict[str, Any]]:
    if not OUTPUT_DIR_PATH.exists():
        return []

    files = []
    for f in sorted(
        OUTPUT_DIR_PATH.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if f.is_file():
            stat = f.stat()
            files.append(
                {"name": f.name, "size": stat.st_size, "modified": stat.st_mtime}
            )

    return files


@app.get("/api/files/{name}")
def get_file(name: str) -> FileResponse:
    base = OUTPUT_DIR_PATH.resolve()
    path = (base / name).resolve()

    if base not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")

    return FileResponse(path)


def main() -> None:
    import uvicorn

    host = "127.0.0.1"
    port = 8765

    threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

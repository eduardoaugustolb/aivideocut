# pyright: basic
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from rich.console import Console

from aivideocut.utils import SRTStringWriter, write_str_to_file

load_dotenv()

console = Console(highlight=False, style="cyan")
rprint = console.print


def transcribe_via_api(
    input_file: Path,
    *,
    language: str = "pt",
    vad: bool = True,
    word_timestamps: bool = True,
    timeout: int = 3600,
) -> dict[str, Any]:
    api_url = os.environ["WHISPER_API_URL"].rstrip("/")
    api_token = os.environ["WHISPER_API_TOKEN"]

    rprint(f"🎙️ Transcribing via Whisper API: {input_file.name}")

    with input_file.open("rb") as f:
        response = requests.post(
            f"{api_url}/transcribe",
            headers={"Authorization": f"Bearer {api_token}"},
            files={"file": (input_file.name, f)},
            data={
                "language": language,
                "vad": str(vad).lower(),
                "word_timestamps": str(word_timestamps).lower(),
            },
            timeout=timeout,
        )

    response.raise_for_status()
    result = response.json()

    rprint(
        f"✅ Transcription done in {result['processing_time']}s "
        f"({result['segments_count']} segments, {result['words_count']} words)"
    )

    return result


def transcribe_to_srt(
    input_file: Path,
    output_path: Path,
    *,
    language: str = "pt",
    dry_run: bool = False,
) -> Path:
    if dry_run:
        rprint(f"🎙️ [dry-run] Would transcribe {input_file.name} -> {output_path}")
        return output_path

    result = transcribe_via_api(input_file, language=language)

    srt_content = SRTStringWriter().write_result(result)
    write_str_to_file(srt_content, path=output_path, create_parents=True)

    rprint(f"✅ Saved transcription to: {output_path}")

    return output_path


def enhance_via_api(
    input_file: Path,
    output_path: Path,
    *,
    denoise_only: bool = False,
    nfe: int = 64,
    lambd: float = 0.9,
    tau: float = 0.5,
    solver: str = "midpoint",
    dry_run: bool = False,
    timeout: int = 3600,
) -> Path:
    if dry_run:
        rprint(f"🎚️ [dry-run] Would enhance {input_file.name} -> {output_path}")
        return output_path

    api_url = os.environ["WHISPER_API_URL"].rstrip("/")
    api_token = os.environ["WHISPER_API_TOKEN"]

    rprint(f"🎚️ Enhancing via Speech API: {input_file.name}")

    with input_file.open("rb") as f:
        response = requests.post(
            f"{api_url}/enhance",
            headers={"Authorization": f"Bearer {api_token}"},
            files={"file": (input_file.name, f)},
            data={
                "denoise_only": str(denoise_only).lower(),
                "nfe": str(nfe),
                "lambd": str(lambd),
                "tau": str(tau),
                "solver": solver,
            },
            timeout=timeout,
        )

    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    rprint(f"✅ Saved enhanced audio to: {output_path}")

    return output_path

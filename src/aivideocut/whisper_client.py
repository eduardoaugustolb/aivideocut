# pyright: basic
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from rich.console import Console

from aivideocut.utils import SRTStringWriter, write_str_to_file

load_dotenv()

console = Console(highlight=False, style="cyan")
rprint = console.print


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text

    detail = payload.get("detail", payload)
    if isinstance(detail, dict):
        parts = [f"{detail.get('error', 'Error')}: {detail.get('message', '')}"]
        if detail.get("type"):
            parts.append(f"type: {detail['type']}")
        if detail.get("traceback"):
            tb_lines = detail["traceback"].strip().splitlines()
            parts.append("\n".join(tb_lines[-12:]))
        return "\n".join(parts)
    return str(detail)


def _post_with_retry(
    url: str,
    *,
    input_file: Path | None = None,
    file_bytes: bytes | None = None,
    filename: str = "file",
    api_token: str,
    data: dict[str, str],
    retries: int = 4,
    timeout: int = 3600,
) -> requests.Response:
    import io

    headers = {"Authorization": f"Bearer {api_token}"}
    last_detail = ""
    for attempt in range(1, retries + 1):
        try:
            with input_file.open("rb") if input_file is not None else io.BytesIO(file_bytes) as f:
                fname = input_file.name if input_file is not None else filename
                response = requests.post(
                    url,
                    headers=headers,
                    files={"file": (fname, f)},
                    data=data,
                    timeout=timeout,
                )
        except requests.RequestException as exc:
            last_detail = str(exc)
            response = None
        else:
            if response.ok:
                return response
            last_detail = _error_detail(response)
            if response.status_code not in (408, 409, 422, 429, 500, 502, 503, 504):
                break

        if attempt < retries:
            delay = 15 * attempt
            rprint(f"🔁 Retry {attempt}/{retries} in {delay}s — {last_detail}")
            time.sleep(delay)

    raise requests.exceptions.HTTPError(
        f"Failed after {retries} attempts ({url}): {last_detail}"
    )


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

    response = _post_with_retry(
        f"{api_url}/transcribe",
        input_file=input_file,
        api_token=api_token,
        data={
            "language": language,
            "vad": str(vad).lower(),
            "word_timestamps": str(word_timestamps).lower(),
        },
        timeout=timeout,
    )

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
    atten_lim_db: float | None = None,
    dry_run: bool = False,
    retries: int = 4,
    timeout: int = 3600,
    chunk_seconds: int = 120,
) -> Path:
    if dry_run:
        rprint(f"🎚️ [dry-run] Would enhance {input_file.name} -> {output_path}")
        return output_path

    api_url = os.environ["WHISPER_API_URL"].rstrip("/")
    api_token = os.environ["WHISPER_API_TOKEN"]

    def _enhance_one(file_bytes: bytes, filename: str) -> bytes:
        data: dict[str, str] = {
            "denoise_only": str(denoise_only).lower(),
        }
        if atten_lim_db is not None:
            data["atten_lim_db"] = str(atten_lim_db)
        response = _post_with_retry(
            f"{api_url}/enhance",
            file_bytes=file_bytes,
            filename=filename,
            api_token=api_token,
            data=data,
            retries=retries,
            timeout=timeout,
        )
        return response.content

    if chunk_seconds > 0:
        import io
        import tempfile

        import numpy as np
        import soundfile as sf

        data, orig_sr = sf.read(input_file, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        duration_samples = int(orig_sr * chunk_seconds)
        total = data.shape[0]
        n_chunks = -(-total // duration_samples)

        if n_chunks > 1:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            enhanced_chunks: list[np.ndarray] = []
            new_sr = None

            for i in range(n_chunks):
                start = i * duration_samples
                end = min(total, (i + 1) * duration_samples)
                raw = data[start:end]
                rprint(f"🎚️  Chunk {i + 1}/{n_chunks}: enhancing {raw.shape[0] / orig_sr:.1f}s")

                buf = io.BytesIO()
                sf.write(buf, raw, orig_sr, format="FLAC")
                chunk_bytes = _enhance_one(buf.getvalue(), f"{input_file.stem}_chunk{i}.flac")

                with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
                    tmp.write(chunk_bytes)
                    tmp_path = tmp.name
                try:
                    enhanced, srn = sf.read(tmp_path, dtype="float32")
                finally:
                    os.remove(tmp_path)
                new_sr = srn
                enhanced_chunks.append(enhanced)

            result = np.concatenate(enhanced_chunks)
            sf.write(output_path, result, new_sr or orig_sr, format="FLAC")
            rprint(f"✅ Saved enhanced audio ({n_chunks} chunks) to: {output_path}")
            return output_path

    rprint(f"🎚️ Enhancing via Speech API: {input_file.name}")

    data: dict[str, str] = {
        "denoise_only": str(denoise_only).lower(),
    }
    if atten_lim_db is not None:
        data["atten_lim_db"] = str(atten_lim_db)

    response = _post_with_retry(
        f"{api_url}/enhance",
        input_file=input_file,
        api_token=api_token,
        data=data,
        retries=retries,
        timeout=timeout,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    rprint(f"✅ Saved enhanced audio to: {output_path}")

    return output_path

# pyright: basic
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

from rich import print as rprint

from aivideocut.ai_client import ask_ai
from aivideocut.utils import write_str_to_file


class TextResponse(Protocol):
    text: str | None
    model_version: str


def run_gemini_chunked_pipeline(
    *,
    chunks: Iterable[str],
    prompt_fn: Callable[[str], str],
    output_path: Path,
    chunk_separator: str = "\n\n",
    dry_run: bool = False,
    ask_fn: Callable[[str], TextResponse] = ask_ai,
) -> str:
    response_text = ""
    model_version = ""

    for chunk in chunks:
        prompt = prompt_fn(chunk)

        if dry_run:
            rprint(prompt, "\n\n---\n\n")
            continue

        ai_response = ask_fn(prompt)
        ai_response_text = ai_response.text

        if ai_response_text:
            response_text += ai_response_text.strip()
            response_text += chunk_separator
            model_version = ai_response.model_version

        rprint(response_text)

    if dry_run:
        return response_text

    if not response_text:
        rprint("\n🔴 AI did not return the text")
        return response_text

    rprint("\n\n")
    rprint(response_text)

    write_str_to_file(response_text, path=output_path, create_parents=True)
    rprint(f"\n✅ Saved to: {output_path.name} (model = {model_version})")

    return response_text

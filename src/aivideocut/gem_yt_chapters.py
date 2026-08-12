# pyright: basic

from functools import partial

from aivideocut.configs import (
    CHAPTERS_YT_FILE_PATH,
    ORIGINAL_SRT_FILE_PATH,
    OUTPUT_DIR_PATH,
    PROMPT_MAX_CHARS,
)
from aivideocut.gem_pipeline import run_gemini_chunked_pipeline
from aivideocut.gem_prompts import create_youtube_chapters_prompt
from aivideocut.utils import create_file_path, read_file_path, split_srt_blocks


def gem_yt_chapters(*, dry_run: bool = False) -> None:
    srt_content = read_file_path(ORIGINAL_SRT_FILE_PATH)
    srt_blocks = split_srt_blocks(srt_content, max_chars=PROMPT_MAX_CHARS)
    chunks = ("\n\n".join(block) for block in srt_blocks)

    output_path = create_file_path(
        full_filename=CHAPTERS_YT_FILE_PATH.name,
        parent=OUTPUT_DIR_PATH,
    )

    run_gemini_chunked_pipeline(
        chunks=chunks,
        prompt_fn=partial(
            create_youtube_chapters_prompt,
            additional_context=(
                "Vídeo educacional mostrando exemplos de uso avançado de "
                "f-string no Python."
            ),
        ),
        output_path=output_path,
        chunk_separator="\n",
        dry_run=dry_run,
    )


if __name__ == "__main__":
    gem_yt_chapters(dry_run=False)

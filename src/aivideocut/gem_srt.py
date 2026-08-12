# pyright: basic

from functools import partial

from aivideocut.configs import (
    ORIGINAL_SRT_FILE_PATH,
    OUTPUT_DIR_PATH,
    PROMPT_MAX_CHARS,
    SRT_FIXED_FILENAME,
)
from aivideocut.gem_pipeline import run_gemini_chunked_pipeline
from aivideocut.gem_prompts import create_fix_srt_prompt
from aivideocut.utils import create_file_path, read_file_path, split_srt_blocks


def fix_srt_typos() -> None:
    srt_content = read_file_path(ORIGINAL_SRT_FILE_PATH)
    srt_blocks = split_srt_blocks(srt_content, max_chars=PROMPT_MAX_CHARS)
    chunks = ("\n\n".join(block) for block in srt_blocks)

    output_path = create_file_path(
        full_filename=SRT_FIXED_FILENAME,
        parent=OUTPUT_DIR_PATH,
    )

    run_gemini_chunked_pipeline(
        chunks=chunks,
        prompt_fn=partial(
            create_fix_srt_prompt,
            additional_context="Aula educacional sobre programação",
        ),
        output_path=output_path,
    )


if __name__ == "__main__":
    fix_srt_typos()

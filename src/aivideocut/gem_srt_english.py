# pyright: basic

from functools import partial

from aivideocut.configs import (
    OUTPUT_DIR_PATH,
    PROMPT_MAX_CHARS,
    SRT_FIXED_ENGLISH_FILENAME,
    SRT_FIXED_FILENAME,
)
from aivideocut.gem_pipeline import run_gemini_chunked_pipeline
from aivideocut.gem_prompts import create_translate_srt_pt_to_en_prompt
from aivideocut.utils import create_file_path, read_file_path, split_srt_blocks


def gem_translate_srt_to_en() -> None:
    fixed_srt_path = create_file_path(
        full_filename=SRT_FIXED_FILENAME,
        parent=OUTPUT_DIR_PATH,
    )
    srt_content = read_file_path(fixed_srt_path)
    srt_blocks = split_srt_blocks(srt_content, max_chars=PROMPT_MAX_CHARS)
    chunks = ("\n\n".join(block) for block in srt_blocks)

    output_path = create_file_path(
        full_filename=SRT_FIXED_ENGLISH_FILENAME,
        parent=OUTPUT_DIR_PATH,
    )

    run_gemini_chunked_pipeline(
        chunks=chunks,
        prompt_fn=partial(
            create_translate_srt_pt_to_en_prompt,
            additional_context="Aula educacional sobre programação",
        ),
        output_path=output_path,
    )


if __name__ == "__main__":
    gem_translate_srt_to_en()

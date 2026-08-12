# pyright: basic

from functools import partial

from aivideocut.configs import (
    OUTPUT_DIR_PATH,
    PROMPT_MAX_CHARS,
    SRT_FIXED_FILENAME,
    SUMMARY_FILE_PATH,
)
from aivideocut.gem_pipeline import run_gemini_chunked_pipeline
from aivideocut.gem_prompts import create_summary_prompt
from aivideocut.utils import (
    create_file_path,
    extract_text_from_srt,
    read_file_path,
    smart_text_split,
)


def generate_summary(*, dry_run: bool = False) -> None:
    fixed_srt_path = create_file_path(
        full_filename=SRT_FIXED_FILENAME,
        parent=OUTPUT_DIR_PATH,
    )
    srt_content = read_file_path(fixed_srt_path)
    extracted_srt_text = extract_text_from_srt(srt_content)

    chunks = smart_text_split(extracted_srt_text, approx_max_chars=PROMPT_MAX_CHARS)

    output_path = create_file_path(
        full_filename=SUMMARY_FILE_PATH.name,
        parent=OUTPUT_DIR_PATH,
    )

    run_gemini_chunked_pipeline(
        chunks=chunks,
        prompt_fn=partial(
            create_summary_prompt,
            additional_context=(
                "Vídeo educacional mostrando exemplos de uso avançado de "
                "f-string no Python."
            ),
        ),
        output_path=output_path,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    generate_summary()

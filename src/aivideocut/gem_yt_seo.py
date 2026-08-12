# pyright: basic

from functools import partial

from aivideocut.configs import (
    OUTPUT_DIR_PATH,
    SEO_YT_FILE_PATH,
    SUMMARY_FILE_PATH,
)
from aivideocut.gem_pipeline import run_gemini_chunked_pipeline
from aivideocut.gem_prompts import create_youtube_seo_prompt
from aivideocut.utils import create_file_path, read_file_path


def gem_yt_seo() -> None:
    input_path = create_file_path(
        full_filename=SUMMARY_FILE_PATH.name,
        parent=OUTPUT_DIR_PATH,
    )
    summary = read_file_path(input_path)

    output_path = create_file_path(
        full_filename=SEO_YT_FILE_PATH.name,
        parent=OUTPUT_DIR_PATH,
    )

    run_gemini_chunked_pipeline(
        chunks=[summary],
        prompt_fn=partial(
            create_youtube_seo_prompt,
            additional_context=(
                "Vídeo educacional mostrando exemplos de uso avançado de "
                "f-string no Python."
            ),
        ),
        output_path=output_path,
        chunk_separator="",
    )


if __name__ == "__main__":
    gem_yt_seo()

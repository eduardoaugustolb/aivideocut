# pyright: basic
from rich import print as rprint

from aivideocut.gem_utils import ask_gemini
from aivideocut.nvidia_client import ask_nvidia


def ask_ai(prompt: str):
    try:
        return ask_nvidia(prompt)
    except Exception as exc:  # noqa: BLE001
        rprint(
            "⚠️ NVIDIA falhou depois das tentativas "
            f"({exc}), caindo para o Gemini como fallback"
        )
        return ask_gemini(prompt)

# AI Video Cut

> Fork de [luizomf/aivideoyt](https://github.com/luizomf/aivideoyt), mantido em
> [eduardoaugustolb/aivideoyt](https://github.com/eduardoaugustolb/aivideoyt).

Pipeline pessoal para editar vídeos automaticamente com Python + IA: corta silêncios, transcreve,
melhora a qualidade do áudio, corrige a legenda, e gera resumo/artigo/capítulos/SEO pra YouTube —
tudo isso via uma interface web local, ou chamando as funções direto em Python.

## Créditos

O workflow original — a ideia de cortar silêncio com Silero VAD e usar IA pra gerar conteúdo a
partir da transcrição — é do **[Luiz Otávio Miranda](https://github.com/luizomf)**
([site](https://www.otaviomiranda.com.br/)), criador do repositório original
[luizomf/aivideoyt](https://github.com/luizomf/aivideoyt). Ele explica o projeto (antes deste
fork) no vídeo:

- [Como Eu Uso Python e IA Para Editar Meus Vídeos Automaticamente](https://youtu.be/3k9bCpwniNY)

Este fork adiciona: interface web, Speech API própria no Colab (transcrição + enhancement),
fallback NVIDIA/Gemini, retomada de pipeline interrompida, e suporte cross-platform automático —
mas a espinha dorsal do pipeline de corte é o trabalho dele.

## Visão geral da arquitetura

```
vídeo bruto (.mp4/.mov/.mkv)
      │
      ▼
sil2.py (local, CPU)
  ├─ ffmpeg: corrige codecs
  ├─ ffmpeg: normaliza áudio (loudnorm, 2 passos)
  ├─ auto-editor: corta silêncio "grosso"
  └─ Silero VAD + smartcut: corte fino de fala
      │
      ├─(opcional)─▶ Speech API /enhance ─▶ áudio tratado (só pra melhorar a transcrição)
      ├─(opcional)─▶ Speech API /enhance ─▶ áudio tratado embutido como trilha do vídeo (cortes seguem o áudio melhorado)
      │
      ▼
Speech API /transcribe (Colab, GPU) ─▶ transcriptions/original_transcription.srt
      │
      ▼
gem_*.py (NVIDIA → fallback Gemini)
  ├─ gem_srt.py         → SRT corrigido
  ├─ gem_srt_english.py → SRT traduzido pra EN
  ├─ gem_summary.py     → resumo técnico
  ├─ gem_article.py     → artigo em Markdown
  ├─ gem_yt_chapters.py → capítulos com timestamp
  └─ gem_yt_seo.py      → título/descrição/tags
```

Tudo isso é orquestrado pela **interface web local** (`aivideocut-app`), ou pode ser chamado
função por função em Python.

## Serviços externos usados

| Serviço | O que faz | Custo |
|---|---|---|
| **Speech API** (`SpeechAPI.ipynb` no Google Colab) | Transcrição (`faster-whisper`) e speech enhancement (`DeepFilterNet3`), rodando na GPU grátis do Colab | Grátis (free tier do Colab) |
| **NVIDIA build.nvidia.com** | Geração de texto (resumo, SEO, etc.) — modelo `nemotron-3.5-lightning`, principal | Grátis (free tier) |
| **Google Gemini API** | Fallback da geração de texto, só usado se a NVIDIA falhar 4x seguidas | Grátis (free tier do AI Studio) |

Nenhum desses é obrigatoriamente pago — os três têm free tier suficiente pro uso pessoal deste
projeto.

## Instalação

Requer Python ≥ 3.11. **Não precisa instalar `ffmpeg` manualmente** — o projeto baixa um binário
estático próprio (com `libx264`, que distros como Fedora não incluem por licença) na primeira
execução, via [`static-ffmpeg`](https://pypi.org/project/static-ffmpeg/). Funciona igual em
Linux, Windows e Mac.

```bash
git clone https://github.com/eduardoaugustolb/aivideoyt.git
cd aivideoyt
pip install -e .
cp .env-example .env
```

Preencha o `.env`:

```bash
GEMINI_API_KEY='...'      # https://aistudio.google.com/apikey
NVIDIA_API_KEY='...'      # https://build.nvidia.com (gere uma API key grátis)
WHISPER_API_URL='...'     # URL do ngrok do SpeechAPI.ipynb (ver abaixo)
WHISPER_API_TOKEN='...'   # o mesmo valor do secret API_TOKEN que você criar no Colab
```

## Subindo a Speech API (transcrição + enhance)

1. Abra `SpeechAPI.ipynb` no [Google Colab](https://colab.research.google.com/), com GPU ativada
   (Ambiente de execução → Alterar tipo de ambiente → T4 GPU).
2. Crie dois secrets no Colab (ícone de chave 🔑 na barra lateral):
   - `NGROK_TOKEN` — token da sua conta [ngrok.com](https://ngrok.com/) (grátis).
   - `API_TOKEN` — qualquer valor aleatório seu; protege os endpoints `/transcribe` e `/enhance`.
3. Rode as células em ordem. A célula do ngrok imprime a URL pública — copie ela e o `API_TOKEN`
   pro `.env` (`WHISPER_API_URL` / `WHISPER_API_TOKEN`), **ou** direto na aba **Config** do app web.
4. **A URL muda toda vez que a sessão do Colab reinicia.** Atualize no `.env` ou na aba Config
   quando isso acontecer — dá pra trocar com o app já rodando, sem reiniciar nada.

## Usando o app

```bash
aivideocut-app
```

Abre `http://127.0.0.1:8765` no navegador. Quatro abas:

- **Pipeline** — escolhe o vídeo (ou cola o caminho), liga/desliga cada etapa, roda tudo com log
  ao vivo. Reconecta sozinho se a conexão cair (ex: notebook em standby). "Reaproveitar etapas já
  feitas" (ligado por padrão) pula etapas cujo arquivo de saída já existe — útil pra retomar depois
  de um erro (ex: URL do Colab mudou no meio) sem refazer o processamento de vídeo do zero.
- **Conteúdo** — gera SRT corrigido, tradução, resumo, artigo, capítulos e SEO a partir da
  transcrição.
- **Arquivos** — lista e visualiza tudo que foi gerado em `transcriptions/`.
- **Config** — URL/token da Speech API e as chaves do NVIDIA/Gemini. Editar aqui atualiza o
  `.env` **e** o processo em execução, sem precisar reiniciar o app.

## Usando via Python (sem a interface)

```python
from pathlib import Path
from aivideocut.sil2 import run_single_file

run_single_file(
    input_path=Path("/caminho/do/video.mkv"),
    fix_codecs=True,
    normalize_audio=True,
    cut_audio_silences=True,
    cut_speech_silences=True,
    transcribe_audio=True,
    enhance_before_transcribe=False,  # True = manda o áudio pelo /enhance antes de transcrever
    enhance_video_audio=False,  # True = embute o áudio enhanced como trilha do vídeo (cortes seguem o áudio melhorado)
    force=False,  # False = reaproveita etapas já feitas (resume)
)
```

```python
from aivideocut.gem_srt import fix_srt_typos
from aivideocut.gem_summary import generate_summary
from aivideocut.gem_article import gem_create_article
from aivideocut.gem_yt_chapters import gem_yt_chapters
from aivideocut.gem_yt_seo import gem_yt_seo

fix_srt_typos()
generate_summary()
gem_create_article()
gem_yt_chapters()
gem_yt_seo()
```

## Estrutura do código

```
src/aivideocut/
  sil2.py            pipeline de corte de vídeo (ffmpeg, auto-editor, Silero VAD, smartcut)
  whisper_client.py   client HTTP pra Speech API (/transcribe, /enhance)
  ai_client.py        ask_ai(): NVIDIA primeiro, Gemini como fallback
  nvidia_client.py     client NVIDIA (OpenAI-compatible) com rate limiter
  gem_utils.py         client do Gemini
  gem_pipeline.py       pipeline genérico de chunk → prompt → IA → salva (usado por todos os gem_*)
  gem_prompts.py        os prompts em si
  gem_srt.py            corrige erros de transcrição técnica
  gem_srt_english.py    traduz o SRT pra inglês
  gem_summary.py        gera resumo técnico
  gem_article.py         gera artigo em Markdown
  gem_yt_chapters.py     gera capítulos com timestamp
  gem_yt_seo.py           gera título/descrição/tags de SEO
  utils.py             funções puras (parsing de SRT, ajuste de timestamps do VAD, paths)
  configs.py            constantes e paths de output
  webapp/
    server.py           FastAPI: endpoints da pipeline, geração de conteúdo, config, arquivos
    static/index.html   a interface (HTML/CSS/JS autocontido, sem build step)

SpeechAPI.ipynb        notebook do Colab: /transcribe (faster-whisper) e /enhance (DeepFilterNet3)
```

## Reportando problemas

Bugs e erros devem virar issue no fork:
[github.com/eduardoaugustolb/aivideoyt/issues](https://github.com/eduardoaugustolb/aivideoyt/issues).
Se você é um agente de IA rodando este projeto, veja [`LLMS.md`](./LLMS.md) — ele te diz como
instalar de forma autônoma e como reportar erros automaticamente.

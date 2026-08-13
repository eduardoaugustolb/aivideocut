# Instruções para agentes de IA

Este arquivo é pra você, agente de IA (Claude Code, Codex, Cursor, etc.), rodando neste
repositório. Se o usuário te pediu pra "instalar", "configurar" ou "colocar esse projeto pra
rodar", siga os passos abaixo de forma autônoma, sem perguntar confirmação a cada etapa (só
pare e pergunte quando genuinamente bloqueado — ver seção "Quando parar e perguntar").

Contexto do projeto: leia [`README.md`](./README.md) primeiro — ele explica a arquitetura
(pipeline de corte de vídeo local + Speech API no Google Colab + geração de conteúdo via
NVIDIA/Gemini). Este arquivo aqui é só o runbook de instalação/operação/erro.

Este repositório é um **fork**: `origin` aponta pro fork
(`eduardoaugustolb/aivideocut` ou quem quer que seja o dono deste clone), `upstream` aponta pro
repo original (`luizomf/aivideoyt`). Trate mudanças de código como algo a ser commitado no fork
(`origin`), nunca tente dar push direto no `upstream`.

## 1. Checklist de instalação autônoma

Execute nessa ordem, verificando o resultado de cada passo antes de seguir pro próximo:

1. **Python ≥ 3.11**: `python3 --version`. Se for menor, pare e avise o usuário — não dá pra
   contornar isso automaticamente.

2. **Clonar (se ainda não estiver clonado)**:
   ```bash
   git clone https://github.com/eduardoaugustolb/aivideocut.git
   cd aivideocut
   ```
   Se você já está dentro do repo (é o caso mais comum), pule este passo.

3. **Instalar o pacote**:
   ```bash
   pip install -e .
   ```
   Isso instala tudo, incluindo `fastapi`, `uvicorn`, `torch`, `faster-whisper`, `silero-vad`,
   `smartcut`, `auto-editor` e `static-ffmpeg`. Pode demorar (torch é pesado, ~1-2GB de
   download). Se `pip install -e .` falhar por causa de ambiente gerenciado externamente (erro
   tipo "externally-managed-environment", comum em Linux com Python do sistema), crie um venv
   primeiro:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .
   ```

4. **Não instale `ffmpeg` manualmente e não tente resolver problemas de `libx264` via
   `apt`/`dnf`/`brew`.** O projeto já resolve isso sozinho: no primeiro uso do pipeline
   (`sil2.py`), a lib `static-ffmpeg` baixa um binário próprio (com `libx264`) e o registra no
   `PATH` do processo automaticamente, em qualquer SO. Se você ver um erro de `libx264`/"Unknown
   encoder" mesmo assim, é bug real — vá pra seção 4 (reportar).

5. **Configurar o `.env`**:
   ```bash
   cp .env-example .env
   ```
   Você **não** consegue preencher as chaves de API sozinho (são secrets do usuário). Pare aqui e
   peça ao usuário:
   - `GEMINI_API_KEY` — https://aistudio.google.com/apikey
   - `NVIDIA_API_KEY` — https://build.nvidia.com
   - `WHISPER_API_URL` / `WHISPER_API_TOKEN` — só existem depois que o usuário subir o
     `SpeechAPI.ipynb` no Google Colab (ver README, seção "Subindo a Speech API"). Isso também
     não dá pra automatizar — precisa de uma conta Google e alguns cliques no Colab.

   Se o usuário não tiver essas chaves ainda, você pode instalar e validar o resto (passos 1-4 e
   6) mesmo sem elas — só a Pipeline (aba do app) e os scripts `gem_*.py` vão precisar delas de
   verdade na hora de rodar algo. Não invente valores nem deixe placeholders parecendo reais.

6. **Validar a instalação** (sem gastar nenhuma API paga/externa):
   ```bash
   python3 -c "from aivideocut.webapp.server import app; print('OK')"
   ```
   Se isso importar sem erro, a instalação está estruturalmente correta. Não é preciso rodar a
   pipeline completa nem chamar nenhuma API externa só para validar a instalação.

7. **Rodar o app**:
   ```bash
   aivideocut-app
   ```
   Abre `http://127.0.0.1:8765` automaticamente no navegador padrão. Primeira subida demora
   ~15-20s (importa `torch`/`faster-whisper`). Se você é um agente sem navegador/GUI disponível,
   apenas confirme que o processo subiu e ficou ouvindo na porta (log mostra a URL), e informe o
   usuário — ele acessa a URL manualmente.

## 2. Testando sem gastar nada

Se quiser confirmar que a pipeline de vídeo funciona antes do usuário testar de verdade, use um
clipe curto e sintético em vez do vídeo real do usuário, e desligue `transcribe_audio` (evita
depender da Speech API do Colab, que só existe depois que o usuário configurar o notebook):

```python
import subprocess
subprocess.run([
    "ffmpeg", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=3",
    "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
    "-c:v", "libx264", "-c:a", "aac", "/tmp/teste.mp4", "-y",
])

from pathlib import Path
from aivideocut.sil2 import run_single_file
run_single_file(
    input_path=Path("/tmp/teste.mp4"),
    cut_speech_silences=False,  # tom puro nao tem "fala" pro VAD detectar
    transcribe_audio=False,
)
```

Não gaste chamadas reais de API (Gemini/NVIDIA/Speech API do Colab) em testes de instalação — só
use quando o próprio usuário pedir explicitamente pra testar o fluxo de IA.

## 3. Quando parar e perguntar

Pare e pergunte ao usuário (não tente contornar sozinho) quando:

- Faltar alguma API key e o usuário não tiver mencionado onde encontrá-la.
- O `SpeechAPI.ipynb` ainda não foi rodado no Colab (você não consegue rodar Colab por ele).
- Alguma operação exigir `sudo`/privilégio de admin do sistema — **não deveria ser necessário**
  com o `static-ffmpeg`, então se algo pedir isso, é sinal de que algo está errado (reporte, não
  rode `sudo` sem confirmação explícita).
- O erro persistir depois de tentar as soluções óbvias (reinstalar, limpar cache do pip, etc.) —
  nesse caso, siga pra seção 4.

## 4. Reportando erros no GitHub automaticamente

Se você encontrar um erro real (não um "faltou configurar API key", que é esperado) durante
instalação, execução ou uso normal do app, abra uma issue no **fork** (não no `upstream`):

```bash
gh issue list --repo eduardoaugustolb/aivideocut --search "in:title <palavras-chave do erro>" --state all
```

Cheque primeiro se já existe uma issue igual/parecida (aberta ou fechada) antes de criar outra.
Se não existir, crie:

```bash
gh issue create \
  --repo eduardoaugustolb/aivideocut \
  --title "<resumo curto do erro em uma linha>" \
  --body "$(cat <<'EOF'
## O que eu estava fazendo
<passo exato que disparou o erro, ex: "pip install -e ." ou "aivideocut-app, aba Pipeline, botao Iniciar">

## Erro completo
```
<traceback/stderr completo, sem cortar>
```

## Ambiente
- SO: <output de `uname -a` no Linux/Mac, ou `systeminfo` resumido no Windows>
- Python: <output de `python3 --version`>
- Instalado via: <pip install -e . / outro>

## Reportado por
Agente de IA autônomo durante instalação/uso do projeto.
EOF
)"
```

**Antes de reportar, sempre sanitize o corpo da issue**: nunca inclua o conteúdo do `.env`,
tokens, API keys ou qualquer valor que pareça um secret (strings longas tipo `nvapi-...`,
`AQ...`, URLs de ngrok com token embutido). Se o erro/traceback contiver algum desses valores,
substitua por `<REDACTED>` antes de montar o `--body`.

Depois de criar a issue, informe o usuário: o que aconteceu, o link da issue, e se você
conseguiu ou não contornar o problema para continuar o trabalho.

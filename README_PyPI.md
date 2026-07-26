# stream-translator-gpt

Real-time transcription and translation for live streams, local media files, and device audio. Available as a command-line tool and a Gradio WebUI.

Full documentation: [GitHub](https://github.com/ionic-bond/stream-translator-gpt)

## Quick Start on Colab (Recommended)

The easiest way to use this tool — no local environment to set up, and Colab's performance is more than enough for stable everyday use. All you need is your own API key, depending on which services you use:

- [Create a Google API key](https://aistudio.google.com/app/apikey) for **Gemini API** translation — recommended, since the Gemini Flash-Lite model has a free quota of 15 requests per minute / 500 per day
- [Create an OpenAI API key](https://platform.openai.com/api-keys) for **OpenAI Transcription API** transcription or **GPT API** translation (any OpenAI-compatible API can also be used)

| Command Line | WebUI |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/stream_translator.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/webui.ipynb) |

(Due to frequent scraping and theft of API keys, we are unable to provide a trial API key. You need to fill in your own API key.)

## Local Installation (Advanced)

Running locally requires some experience with Python environments (especially on Windows). If in doubt, use Colab instead.

1. **Python** >= 3.10
2. **FFmpeg** (skip if already installed):
   - Windows: `winget install ffmpeg`
   - Linux (Debian/Ubuntu): `sudo apt install ffmpeg`
3. For **local transcription** (Whisper / Faster-Whisper / SimulStreaming / HuggingFace ASR) — not needed if you only use the OpenAI Transcription API:
   - [Install CUDA on your system](https://developer.nvidia.com/cuda-downloads)
   - [Install PyTorch (with CUDA) to your Python](https://pytorch.org/get-started/locally/)
   - [Install cuDNN to your CUDA dir](https://developer.nvidia.com/cudnn-downloads) if you want to use **Faster-Whisper**

Then install the package:

```
pip install stream-translator-gpt -U
```

Or with the WebUI included:

```
pip install stream-translator-gpt[webui] -U
```

## Usage

### WebUI

```
stream-translator-gpt-webui
```

Then open the printed local URL in your browser.

### Command line

```
stream-translator-gpt URL [OPTIONS]
```

**Transcription backends** (default is local **Whisper**):

- ```stream-translator-gpt {URL} --language {input_language}```
- **Faster-Whisper**: ```stream-translator-gpt {URL} --language {input_language} --use-faster-whisper```
- **SimulStreaming**: ```stream-translator-gpt {URL} --language {input_language} --use-simul-streaming```
- **SimulStreaming** with **Faster-Whisper** as the encoder: ```stream-translator-gpt {URL} --language {input_language} --use-simul-streaming --use-faster-whisper```
- **OpenAI Transcription API**: ```stream-translator-gpt {URL} --language {input_language} --use-openai-transcription-api --openai-api-key {your_openai_key}```
- **HuggingFace ASR** model (requires `pip install stream-translator-gpt[hf_asr]`): ```stream-translator-gpt {URL} --model {hf_model_name} --use-hf-asr```

**Translation** (enabled by setting `--translation-prompt`; the provider is chosen by which API key you fill in):

- By **Gemini**: ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key}```
- By **GPT**: ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --openai-api-key {your_openai_key}```

**Input sources** (besides stream URLs):

- Local video/audio file: ```stream-translator-gpt /path/to/file --language {input_language}```
- System audio (loopback): ```stream-translator-gpt device --language {input_language}```
- Microphone: ```stream-translator-gpt device --mic --language {input_language}```

**Output destinations** (besides the terminal):

- Discord: ```stream-translator-gpt {URL} --language {input_language} --discord-webhook-url {your_discord_webhook_url}```
- Telegram: ```stream-translator-gpt {URL} --language {input_language} --telegram-token {your_telegram_token} --telegram-chat-id {your_telegram_chat_id}```
- Cqhttp: ```stream-translator-gpt {URL} --language {input_language} --cqhttp-url {your_cqhttp_url} --cqhttp-token {your_cqhttp_token}```
- .srt subtitle file (offline generation): ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key} --no-show-transcribe-result --retry-if-translation-fails --output-timestamps --output-file-path ./result.srt```

See the [GitHub README](https://github.com/ionic-bond/stream-translator-gpt) for the full option reference.

# stream-translator-gpt

Real-time transcription and translation for live streams, local media files, and device audio. Available as a command-line tool and a Gradio WebUI.

## Quick Start on Colab (Recommended)

The easiest way to use this tool, no local environment to set up, and Colab's performance is more than enough for stable everyday use. All you need is your own API key, depending on which services you use:

- [Create a **Google API key**](https://aistudio.google.com/app/apikey) for **Gemini API** translation (recommended, since the **Gemini Flash-Lite** model has a free quota of 15 requests per minute / 500 per day)
- [Create an **OpenAI API key**](https://platform.openai.com/api-keys) for **OpenAI Transcription API** transcription or **GPT API** translation (any **OpenAI-compatible API** can also be used)

|                                                                                       Command Line                                                                                        |                                                                                     WebUI                                                                                     |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/colab/command_line.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/colab/webui.ipynb) |

(Due to frequent scraping and theft of API keys, we are unable to provide a trial API key. You need to fill in your own API key.)

## Local Installation (Advanced)

Running locally requires some experience with Python environments (especially on Windows) — you are essentially rebuilding an environment similar to Colab's. If in doubt, use Colab instead.

1. **Python** >= 3.10
2. **FFmpeg** (skip if already installed):
   - Windows: `winget install ffmpeg`
   - Linux (Debian/Ubuntu): `sudo apt install ffmpeg`
3. For **local transcription** (Whisper / Faster-Whisper / SimulStreaming / HuggingFace ASR) — not needed if you only use the OpenAI Transcription API:
   - [Install **CUDA** on your system](https://developer.nvidia.com/cuda-downloads)
   - [Install **PyTorch** (with CUDA) to your Python](https://pytorch.org/get-started/locally/)
   - [Install **cuDNN** to your CUDA dir](https://developer.nvidia.com/cudnn-downloads) if you want to use **Faster-Whisper**

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

Then open the printed local URL in your browser. All CLI features are available through the interface, and your settings can be saved as presets.

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
- **HuggingFace ASR** model (requires `pip install stream-translator-gpt[hf_asr]`; only models with `pipeline_tag: automatic-speech-recognition` on Hugging Face Hub are supported): ```stream-translator-gpt {URL} --model {hf_model_name} --use-hf-asr```

**Translation** (enabled by setting `--translation-prompt`; the provider is chosen by which API key you fill in):

- By **Gemini**: ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key}```
- By **GPT**: ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --openai-api-key {your_openai_key}```
- **OpenAI Transcription API** and **Gemini** at the same time: ```stream-translator-gpt {URL} --language {input_language} --use-openai-transcription-api --openai-api-key {your_openai_key} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key}```

> [!TIP]
> The translation prompt is passed to the LLM as-is, so it can carry more than just the language pair. If your API quota allows, give the model some context — who the streamer is, what the stream is about, preferred terminology — and it will translate more accurately and fix ASR errors more reliably.

**Input sources** (besides stream URLs):

- **Local video/audio file**: ```stream-translator-gpt /path/to/file --language {input_language}```
- **System audio** (loopback): ```stream-translator-gpt device --language {input_language}```
- **Microphone**: ```stream-translator-gpt device --mic --language {input_language}```

**Output destinations** (besides the terminal):

- **Discord**: ```stream-translator-gpt {URL} --language {input_language} --discord-webhook-url {your_discord_webhook_url}```
- **Telegram**: ```stream-translator-gpt {URL} --language {input_language} --telegram-token {your_telegram_token} --telegram-chat-id {your_telegram_chat_id}```
- **Cqhttp**: ```stream-translator-gpt {URL} --language {input_language} --cqhttp-url {your_cqhttp_url} --cqhttp-token {your_cqhttp_token}```
- **.srt subtitle file** (offline generation): ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key} --no-show-transcribe-result --retry-if-translation-fails --output-timestamps --output-file-path ./result.srt```

## All Options

> [!NOTE]
> All options accept both hyphens and underscores: `--openai-api-key` and `--openai_api_key` are equivalent.
> Every boolean option has an inverse `--no-*` form that turns it off (e.g. `--no-dynamic-vad-threshold` disables `--dynamic-vad-threshold`, which is on by default).

| Option                             | Default Value                  | Description                                                                                                                                                                                                                                                           |
| :--------------------------------- | :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `URL`                              |                                | The URL of the stream. If a local file path is filled in, it will be used as input. If fill in "device", the input will be obtained from your PC device.                                                                                                              |
| **Overall Options**                |
| `--openai-api-key`                 |                                | OpenAI API key if using GPT translation / OpenAI Transcription API. If you have multiple keys, you can separate them with "," and each key will be used in turn.                                                                                                      |
| `--google-api-key`                 |                                | Google API key if using Gemini translation. If you have multiple keys, you can separate them with "," and each key will be used in turn.                                                                                                                              |
| `--openai-base-url`                |                                | Customize the API endpoint of OpenAI (Affects GPT translation & OpenAI Transcription).                                                                                                                                                                                |
| `--google-base-url`                |                                | Customize the API endpoint of Google (Affects Gemini translation).                                                                                                                                                                                                    |
| `--no-verify-ssl`                  |                                | Disable TLS certificate verification for OpenAI / Google API and HuggingFace downloads. Use this when your API endpoint or proxy has a self-signed or invalid certificate. If the base URL host is a bare IP, verification is disabled automatically.                 |
| `--proxy`                          |                                | Used to set the proxy for all --*-proxy options if they are not specifically set. Also sets http_proxy environment variables.                                                                                                                                         |
| **Input Options**                  |
| `--format`                         | ba/wa*                         | Stream format code, this parameter will be passed directly to yt-dlp. You can get the list of available format codes by `yt-dlp {url} -F`                                                                                                                             |
| `--list-format`                    |                                | Print all available formats then exit.                                                                                                                                                                                                                                |
| `--cookies`                        |                                | Used to open member-only stream, this parameter will be passed directly to yt-dlp.                                                                                                                                                                                    |
| `--input-proxy`                    |                                | Use the specified HTTP/HTTPS/SOCKS proxy for yt-dlp, e.g. http://127.0.0.1:7890.                                                                                                                                                                                      |
| `--device-index`                   |                                | The index of the device that needs to be recorded. If not set, the system default recording device will be used.                                                                                                                                                      |
| `--list-devices`                   |                                | Print all audio devices info then exit.                                                                                                                                                                                                                               |
| `--device-recording-interval`      | 0.5                            | The shorter the recording interval, the lower the latency, but it will increase CPU usage. It is recommended to set it between 0.1 and 1.0.                                                                                                                           |
| `--mic`                            |                                | Use microphone instead of system audio (loopback).                                                                                                                                                                                                                    |
| **Audio Slicing Options**          |
| `--min-audio-length`               | 0.5                            | Minimum slice audio length in seconds.                                                                                                                                                                                                                                |
| `--max-audio-length`               | 30.0                           | Maximum slice audio length in seconds.                                                                                                                                                                                                                                |
| `--target-audio-length`            | 5.0                            | When dynamic no speech threshold is enabled (enabled by default), the program will slice the audio as close to this length as possible.                                                                                                                               |
| `--continuous-no-speech-threshold` | 1.0                            | Slice if there is no speech during this number of seconds. If the dynamic no speech threshold is enabled (enabled by default), the actual threshold will be dynamically adjusted based on this value.                                                                 |
| `--no-dynamic-no-speech-threshold` |                                | Disable dynamic no speech threshold (enabled by default).                                                                                                                                                                                                             |
| `--prefix-retention-length`        | 0.5                            | The length of the retention prefix audio during slicing.                                                                                                                                                                                                              |
| `--vad-threshold`                  | 0.35                           | Range 0~1. The higher this value, the stricter the speech judgment. If dynamic VAD threshold is enabled (enabled by default), this threshold will be adjusted dynamically based on this value.                                                                        |
| `--no-dynamic-vad-threshold`       |                                | Disable dynamic VAD threshold (enabled by default).                                                                                                                                                                                                                   |
| **Transcription Options**          |
| `--model`                          | turbo                          | Select Whisper/Faster-Whisper/SimulStreaming model size. See [here](https://github.com/openai/whisper#available-models-and-languages) for available models.                                                                                                           |
| `--language`                       | auto                           | Language spoken in the stream. Default option is to auto detect the spoken language. See [here](https://github.com/openai/whisper#available-models-and-languages) for available languages.                                                                            |
| `--use-faster-whisper`             |                                | Use Faster-Whisper instead of Whisper. If used with --use-simul-streaming, SimulStreaming with Faster-Whisper as the encoder will be used.                                                                                                                            |
| `--use-simul-streaming`            |                                | Use SimulStreaming instead of Whisper. If used with --use-faster-whisper, SimulStreaming with Faster-Whisper as the encoder will be used.                                                                                                                             |
| `--use-openai-transcription-api`   |                                | Use OpenAI Transcription API instead of the original local Whisper.                                                                                                                                                                                                   |
| `--use-hf-asr`                     |                                | Use a HuggingFace ASR model. Use `--model` to specify the model ID. Requires `pip install stream-translator-gpt[hf_asr]`.                                                                                                                                             |
| `--transcription-filters`          | emoji_filter,repetition_filter | Filters apply to transcription results, separated by ",". We provide emoji_filter and repetition_filter.                                                                                                                                                              |
| `--no-language-based-filter`       |                                | Disable the currently provided English, Chinese, and Japanese language filters based on ASR language (enabled by default).                                                                                                                                            |
| `--transcription-initial-prompt`   |                                | General purpose prompt or glossary for transcription. Format: "Word1, Word2, Word3, ...". This text is always included in the prompt passed to the model.                                                                                                             |
| `--no-transcription-context`       |                                | Disable context (previous sentence) propagation in transcription (enabled by default).                                                                                                                                                                                |
| **Translation Options**            |
| `--gpt-model`                      | gpt-5.4-nano                   | OpenAI's GPT model name, gpt-5.4 / gpt-5.4-mini / gpt-5.4-nano / gpt-5.5 / gpt-5.6-luna                                                                                                                                                                               |
| `--gemini-model`                   | gemini-3.5-flash-lite          | Google's Gemini model name, gemini-3-flash-preview / gemini-3.1-flash-lite / gemini-3.5-flash / gemini-3.5-flash-lite / gemini-3.6-flash                                                                                                                              |
| `--translation-prompt`             |                                | If set, will translate the result text to target language via GPT / Gemini API (According to which API key is filled in). Example: "Translate from Japanese to Chinese". Adding context (who the streamer is, what the stream is about) improves translation quality. |
| `--translation-history-size`       | 0                              | The number of previous transcripts sent as context when calling the LLM API. It is recommended to disable context (set to 0) for weaker models.                                                                                                                       |
| `--translation-timeout`            | 10                             | If the GPT / Gemini translation exceeds this number of seconds, the translation will be discarded.                                                                                                                                                                    |
| `--use-json-result`                |                                | Using JSON result in LLM translation for some locally deployed models.                                                                                                                                                                                                |
| `--retry-if-translation-fails`     |                                | Retry when translation times out/fails. Used to generate subtitles offline.                                                                                                                                                                                           |
| `--temperature`                    |                                | GPT/Gemini parameter. Controls output randomness, higher values produce more diverse results.                                                                                                                                                                         |
| `--top-p`                          |                                | GPT/Gemini parameter. Nucleus sampling threshold, only tokens with cumulative probability above this value are considered.                                                                                                                                            |
| `--top-k`                          |                                | Gemini parameter. Limits token selection to the top K most probable candidates.                                                                                                                                                                                       |
| `--prompt-cache-key`               |                                | GPT parameter. If set, enables prompt caching optimization on the API side.                                                                                                                                                                                           |
| `--reasoning-effort`               |                                | GPT parameter. Controls reasoning depth for reasoning models. Options: none / minimal / low / medium / high / xhigh.                                                                                                                                                  |
| `--verbosity`                      |                                | GPT parameter. Controls the verbosity of the response. Options: auto / short / concise / detailed.                                                                                                                                                                    |
| `--service-tier`                   |                                | GPT parameter. Specifies processing priority tier. Options: auto / default / flex / priority.                                                                                                                                                                         |
| `--debug-mode`                     |                                | Enable debug mode. Print messages sent to LLM and usage info after each translation call.                                                                                                                                                                             |
| `--processing-proxy`               |                                | Use the specified HTTP/HTTPS/SOCKS proxy for Whisper/GPT API (Gemini currently doesn't support specifying a proxy within the program), e.g. http://127.0.0.1:7890.                                                                                                    |
| **Output Options**                 |
| `--output-timestamps`              |                                | Output the timestamp of the text when outputting the text.                                                                                                                                                                                                            |
| `--no-show-transcribe-result`      |                                | Hide the transcription result (shown by default).                                                                                                                                                                                                                     |
| `--output-file-path`               |                                | If set, will save the result text to this path.                                                                                                                                                                                                                       |
| `--cqhttp-url`                     |                                | If set, will send the result text to this Cqhttp server.                                                                                                                                                                                                              |
| `--cqhttp-token`                   |                                | Token of Cqhttp, if it is not set on the server side, it does not need to fill in.                                                                                                                                                                                    |
| `--discord-webhook-url`            |                                | If set, will send the result text to this Discord channel.                                                                                                                                                                                                            |
| `--telegram-token`                 |                                | Token of Telegram bot.                                                                                                                                                                                                                                                |
| `--telegram-chat-id`               |                                | If set, will send the result text to this Telegram chat. Needs to be used with --telegram-token.                                                                                                                                                                      |
| `--output-proxy`                   |                                | Use the specified HTTP/HTTPS/SOCKS proxy for Cqhttp/Discord/Telegram, e.g. http://127.0.0.1:7890.                                                                                                                                                                     |

## Contact Me

Telegram: [@ionic_bond](https://t.me/ionic_bond)

## Donate

[PayPal Donate](https://www.paypal.com/donate/?hosted_button_id=D5DRBK9BL6DUA) or [PayPal](https://paypal.me/ionicbond3)

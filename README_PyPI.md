# stream-translator-gpt

stream-translator-gpt is a command-line tool for real-time transcription and translation of live streams. We have now added an easier-to-use WebUI entry point.

Try it on Colab: 

|                                                                                     WebUI                                                                                     |                                                                                       Command Line                                                                                        |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/webui.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/stream_translator.ipynb) |

(Due to frequent scraping and theft of API keys, we are unable to provide a trial API key. You need to fill in your own API key.)

## Prerequisites

**Linux or Windows:**

1. Python >= 3.8 (Recommend >= 3.10)
2. [**Install CUDA on your system**](https://developer.nvidia.com/cuda-downloads).
3. [**Install cuDNN to your CUDA dir**](https://developer.nvidia.com/cudnn-downloads) if you want to use **Faster-Whisper**.
4. [**Install PyTorch (with CUDA) to your Python**](https://pytorch.org/get-started/locally/).
5. [**Create a Google API key**](https://aistudio.google.com/app/apikey) if you want to use **Gemini API** for translation.
6. [**Create a OpenAI API key**](https://platform.openai.com/api-keys) if you want to use **OpenAI Transcription API** for transcription or **GPT API** for translation.

**If you are in Windows, you also need to:**

1. [**Install and add ffmpeg to your PATH.**](https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10#:~:text=Click%20New%20and%20type%20the,Click%20OK%20to%20apply%20changes.)
2. Install [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) and add it to your PATH.


## WebUI

```
pip install stream-translator-gpt[webui] -U
stream-translator-gpt-webui
```

## Command Line

**Install release version from PyPI:**

```
pip install stream-translator-gpt -U
stream-translator-gpt
```

or

**Clone master version code from Github:**

```
git clone https://github.com/ionic-bond/stream-translator-gpt.git
pip install -r ./stream-translator-gpt/requirements.txt -U
python3 ./stream-translator-gpt/stream_translator_gpt/main.py
```

### Usage

The commands on Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/stream_translator.ipynb) are the recommended usage, below are some other commonly used options.

- Transcribe live streaming (default use **Whisper**):

    ```stream-translator-gpt {URL} --model large --language {input_language}```

- Transcribe by **Faster-Whisper**:

    ```stream-translator-gpt {URL} --model large --language {input_language} --use_faster_whisper```

- Transcribe by **SimulStreaming**:

    ```stream-translator-gpt {URL} --model large --language {input_language} --use_simul_streaming```

- Transcribe by **SimulStreaming** with **Faster-Whisper** as the encoder:

    ```stream-translator-gpt {URL} --model large --language {input_language} --use_simul_streaming --use_faster_whisper```

- Transcribe by **OpenAI Transcription API**:

    ```stream-translator-gpt {URL} --language {input_language} --use_openai_transcription_api --openai_api_key {your_openai_key}```

- Translate to other language by **Gemini**:

    ```stream-translator-gpt {URL} --model large --language ja --translation_prompt "Translate from Japanese to Chinese" --google_api_key {your_google_key}```

- Translate to other language by **GPT**:

    ```stream-translator-gpt {URL} --model large --language ja --translation_prompt "Translate from Japanese to Chinese" --openai_api_key {your_openai_key}```

- Using **OpenAI Transcription API** and **Gemini** at the same time:

    ```stream-translator-gpt {URL} --language ja --use_openai_transcription_api --openai_api_key {your_openai_key} --translation_prompt "Translate from Japanese to Chinese" --google_api_key {your_google_key}```

- Local video/audio file as input:

    ```stream-translator-gpt /path/to/file --model large --language {input_language}```

- Computer microphone as input:

    ```stream-translator-gpt device --model large --language {input_language}```
    
    Will use the system's default audio device as input.

    If you want to use another audio input device, `stream-translator-gpt device --list_devices` get device index and then run the CLI with `--device_index {index}`.

    If you want to use the audio output of another program as input, you need to [**enable stereo mix**](https://www.howtogeek.com/39532/how-to-enable-stereo-mix-in-windows-7-to-record-audio/).

- Sending result to Discord:

    ```stream-translator-gpt {URL} --model large --language {input_language} --discord_webhook_url {your_discord_webhook_url}```

- Sending result to Telegram:

    ```stream-translator-gpt {URL} --model large --language {input_language} --telegram_token {your_telegram_token} --telegram_chat_id {your_telegram_chat_id}```

- Sending result to Cqhttp:

    ```stream-translator-gpt {URL} --model large --language {input_language} --cqhttp_url {your_cqhttp_url} --cqhttp_token {your_cqhttp_token}```

- Saving result to a .srt subtitle file:

    ```stream-translator-gpt {URL} --model large --language ja --translation_prompt "Translate from Japanese to Chinese" --google_api_key {your_google_key} --hide_transcribe_result --retry_if_translation_fails --output_timestamps --output_file_path ./result.srt```

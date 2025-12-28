# stream-translator-gpt

[![PyPI version](https://badge.fury.io/py/stream-translator-gpt.svg)](https://badge.fury.io/py/stream-translator-gpt) [![Python Versions](https://img.shields.io/pypi/pyversions/stream-translator-gpt.svg)](https://pypi.org/project/stream-translator-gpt/) [![Downloads](https://static.pepy.tech/badge/stream-translator-gpt)](https://pepy.tech/project/stream-translator-gpt) [![License](https://img.shields.io/github/license/ionic-bond/stream-translator-gpt.svg)](https://github.com/ionic-bond/stream-translator-gpt/blob/main/LICENSE) [![Gradio](https://img.shields.io/badge/WebUI-Gradio-orange)](https://gradio.app)

[English](./README.md) | 中文

stream-translator-gpt 是一个用于实时转录和翻译直播流的命令行工具。我们新增了更易于使用的 WebUI 入口。

在 Colab 上尝试：

|                                                                                     WebUI                                                                                     |                                                                                          命令行                                                                                           |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/webui.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/stream_translator.ipynb) |

（由于 API key 被频繁爬取和盗用，我们无法提供用于试用的 API key。您需要填写自己的 API key。）

## 工作流

```mermaid
flowchart LR
    subgraph ga["`**输入**`"]
        direction LR
        aa("`**FFmpeg**`")
        ab("`**计算机音频设备**`")
        ac("`**yt-dlp**`")
        ad("`**本地媒体文件**`")
        ae("`**直播流**`")
        ac --> aa
        ad --> aa
        ae --> ac
    end
    subgraph gb["`**音频切片**`"]
        direction LR
        ba("`**Silero VAD**`")
    end
    subgraph gc["`**语音转文字**`"]
        direction LR
        ca("`**Whisper**`")
        cb("`**Faster-Whisper**`")
        cc("`**Simul Streaming**`")
        cd("`**OpenAI Transcription API**`")
    end
    subgraph gd["`**翻译**`"]
        direction LR
        da("`**GPT API**`")
        db("`**Gemini API**`")
    end
    subgraph ge["`**输出**`"]
        direction LR
        ea("`**打印到终端**`")
        ee("`**保存到文件**`")
        ec("`**Discord**`")
        ed("`**Telegram**`")
        eb("`**Cqhttp**`")
    end
    aa --> gb
    ab --> gb
    gb ==> gc
    gc ==> gd
    gd ==> ge
````

使用 [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) 从直播流中提取音频数据。

基于 [**Silero-VAD**](https://github.com/snakers4/silero-vad) 的动态阈值音频切片。

在本地使用 [**Whisper**](https://github.com/openai/whisper) / [**Faster-Whisper**](https://github.com/SYSTRAN/faster-whisper) /  [**Simul Streaming**](https://github.com/ufal/SimulStreaming) 或远程调用 [**OpenAI Transcription API**](https://platform.openai.com/docs/guides/speech-to-text) 进行转录。

使用 OpenAI 的 [**GPT API**](https://platform.openai.com/docs/overview) / Google 的 [**Gemini API**](https://ai.google.dev/gemini-api/docs) 进行翻译。

最后，结果可以打印到终端、保存到文件，或通过社交媒体机器人发送到群组。

## 准备工作

**Linux 或 Windows:**

1. Python >= 3.8 (推荐 >= 3.10)
2. [**在您的系统上安装 CUDA**](https://developer.nvidia.com/cuda-downloads)。
3. 如果您想使用 **Faster-Whisper**，[**请将 cuDNN 安装到您的 CUDA 目录**](https://developer.nvidia.com/cudnn-downloads)。
4. [**为您的 Python 安装 PyTorch (CUDA 版本)**](https://pytorch.org/get-started/locally/)。
81. 如果您想使用 **Gemini API** 进行翻译，[**请创建一个 Google API 密钥**](https://aistudio.google.com/app/apikey)。
82. 如果您想使用 **OpenAI Transcription API** 进行语音转文字或使用 **GPT API** 进行翻译，[**请创建一个 OpenAI API 密钥**](https://platform.openai.com/api-keys)。

**如果您在 Windows 上，您还需要：**

1. [**安装 ffmpeg 并将其添加到您的环境变量。**](https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10#:~:text=Click%20New%20and%20type%20the,Click%20OK%20to%20apply%20changes.)
2. 安装 [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) 并将其添加到您的环境变量。


## WebUI

```
pip install stream-translator-gpt[webui] -U
stream-translator-gpt-webui
```

## 命令行

**从 PyPI 安装稳定版本:**

```
pip install stream-translator-gpt -U
stream-translator-gpt
```

或者

**从 Github 下载开发版本代码:**

```
git clone https://github.com/ionic-bond/stream-translator-gpt.git
pip install -r ./stream-translator-gpt/requirements.txt -U
python3 ./stream-translator-gpt/stream_translator_gpt/main.py
```

### 使用方法

Colab上的命令 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/stream_translator.ipynb) 即为推荐的使用方式，以下是一些其他常用选项。

- 转录直播流 (默认使用 **Whisper**):

    ```stream-translator-gpt {网址} --model large --language {输入语言}```

- 使用 **Faster-Whisper** 进行转录:

    ```stream-translator-gpt {网址} --model large --language {输入语言} --use_faster_whisper```

- 使用 **SimulStreaming** 进行转录:

    ```stream-translator-gpt {网址} --model large --language {输入语言} --use_simul_streaming```

- 使用以 **Faster-Whisper** 作为编码器的 **SimulStreaming** 进行转录:

    ```stream-translator-gpt {网址} --model large --language {输入语言} --use_simul_streaming --use_faster_whisper```

- 使用 **OpenAI Transcription API** 进行转录:

    ```stream-translator-gpt {网址} --language {输入语言} --use_openai_transcription_api --openai_api_key {您的 OpenAI 密钥}```

- 使用 **Gemini** 翻译成其他语言:

    ```stream-translator-gpt {网址} --model large --language ja --translation_prompt "翻译以下日语为中文，只输出译文，不要输出原文，在一行内输出" --google_api_key {您的 Google 密钥}```

- 使用 **GPT** 翻译成其他语言:

    ```stream-translator-gpt {网址} --model large --language ja --translation_prompt "翻译以下日语为中文，只输出译文，不要输出原文，在一行内输出" --openai_api_key {您的 OpenAI 密钥}```

- 同时使用 **OpenAI Transcription API** 和 **Gemini**:

    ```stream-translator-gpt {网址} --language ja --use_openai_transcription_api --openai_api_key {您的 OpenAI 密钥} --translation_prompt "翻译以下日语为中文，只输出译文，不要输出原文，在一行内输出" --google_api_key {您的 Google 密钥}```

- 使用本地视频/音频文件作为输入:

    ```stream-translator-gpt {文件路径} --model large --language {输入语言}```

- 使用电脑麦克风作为输入:

    ```stream-translator-gpt device --model large --language {输入语言}```

  将使用系统的默认音频设备作为输入。

  如果您想使用其他音频输入设备，运行 `stream-translator-gpt device --list_devices` 获取设备索引，然后使用 `--device_index {索引}` 运行 CLI。

  如果您想使用其他程序的音频输出作为输入，您需要 [**启用立体声混音**](https://www.howtogeek.com/39532/how-to-enable-stereo-mix-in-windows-7-to-record-audio/)。

- 发送结果到 Discord:

    ```stream-translator-gpt {网址} --model large --language {输入语言} --discord_webhook_url {您的_discord_webhook_网址}```

- 发送结果到 Telegram:

    ```stream-translator-gpt {网址} --model large --language {输入语言} --telegram_token {您的 Telegram 令牌} --telegram_chat_id {您的 Telegram 聊天 id}```

- 发送结果到 Cqhttp:

    ```stream-translator-gpt {网址} --model large --language {输入语言} --cqhttp_url {您的 cqhttp 地址} --cqhttp_token {您的 cqhttp 令牌}```

- 保存结果到 .srt 字幕文件:

    ```stream-translator-gpt {网址} --model large --language ja --translation_prompt "翻译以下日语为中文，只输出译文，不要输出原文，在一行内输出" --google_api_key {您的 Google 密钥} --hide_transcribe_result --retry_if_translation_fails --output_timestamps --output_file_path ./result.srt```

### 所有选项

| 选项                                    | 默认值                         | 描述                                                                                                                                                                      |
| :-------------------------------------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **通用选项**                            |
| `--proxy`                               |                                | 用于设置所有未特别指定的 --*_proxy 的值。也会设置 http_proxy 等环境变量。                                                                                                 |
| `--openai_api_key`                      |                                | 如果使用 GPT 翻译 / Whisper API，则需要 OpenAI API 密钥。如果您有多个密钥，可以用 "," 分隔，每个密钥将轮流使用。                                                          |
| `--google_api_key`                      |                                | 如果使用 Gemini 翻译，则需要 Google API 密钥。如果您有多个密钥，可以用 "," 分隔，每个密钥将轮流使用。                                                                     |
| **输入选项**                            |                                |                                                                                                                                                                           |
| `URL`                                   |                                | 直播流的 URL。如果填入本地文件路径，则会将其用作输入。如果填入 "device"，将从您的 PC 设备获取输入。                                                                       |
| `--format`                              | ba/wa*                         | 码流格式代码，此参数将直接传递给 yt-dlp。您可以通过 `yt-dlp {url} -F` 获取可用格式代码的列表。                                                                            |
| `--list_format`                         |                                | 打印所有可用格式然后退出。                                                                                                                                                |
| `--cookies`                             |                                | 用于打开会员专属直播，此参数将直接传递给 yt-dlp。                                                                                                                         |
| `--input_proxy`                         |                                | 为 yt-dlp 使用指定的 HTTP/HTTPS/SOCKS 代理，例如 http://127.0.0.1:7890。                                                                                                  |
| `--device_index`                        |                                | 需要录制的设备的索引。如果未设置，将使用系统默认的录音设备。                                                                                                              |
| `--list_devices`                        |                                | 打印所有音频设备信息然后退出。                                                                                                                                            |
| `--device_recording_interval`           | 0.5                            | 录制间隔越短，延迟越低，但会增加 CPU 使用率。建议设置在 0.1 和 1.0 之间。                                                                                                 |
| **音频切片选项**                        |                                |                                                                                                                                                                           |
| `--min_audio_length`                    | 0.5                            | 最小音频切片长度（秒）。                                                                                                                                                  |
| `--max_audio_length`                    | 30.0                           | 最大音频切片长度（秒）。                                                                                                                                                  |
| `--target_audio_length`                 | 5.0                            | 当启用动态无语音阈值时（默认启用），程序将尽可能按接近此长度切割音频。                                                                                                    |
| `--continuous_no_speech_threshold`      | 1.0                            | 如果在此秒数内没有语音，则进行切片。如果启用了动态无语音阈值（默认启用），实际阈值将基于此值动态调整。                                                                    |
| `--disable_dynamic_no_speech_threshold` |                                | 设置此标志以禁用动态静音阈值。                                                                                                                                            |
| `--prefix_retention_length`             | 0.5                            | 切片时保留的前缀音频长度。                                                                                                                                                |
| `--vad_threshold`                       | 0.35                           | 范围 0~1。此值越高，语音判断越严格。如果启用了动态 VAD 阈值（默认启用），此阈值将根据输入语音的 VAD 结果动态调整。                                                        |
| `--disable_dynamic_vad_threshold`       |                                | 设置此标志以禁用动态 VAD 阈值。                                                                                                                                           |
| **转录选项**                            |                                |                                                                                                                                                                           |
| `--model`                               | small                          | 选择 Whisper/Faster-Whisper/Simul Streaming 模型大小。可用模型请参见 [此处](https://github.com/openai/whisper#available-models-and-languages)。                           |
| `--language`                            | auto                           | 直播流中的语言。可用语言请参见 [此处](https://github.com/openai/whisper#available-models-and-languages)。                                                                 |
| `--use_faster_whisper`                  |                                | 设置此标志以使用 Faster-Whisper 进行语音转文字，而不是原始的 OpenAI Whisper。如果与 --use_simul_streaming 一起使用，将使用以 Faster-Whisper 作为编码器的 SimulStreaming。 |
| `--use_simul_streaming`                 |                                | 设置此标志以使用 SimulStreaming 进行语音转文字，而不是原始的 OpenAI Whisper。如果与 --use_faster_whisper 一起使用，将使用以 Faster-Whisper 作为编码器的 SimulStreaming。  |
| `--use_openai_transcription_api`        |                                | 设置此标志以使用 OpenAI transcription API，而不是原始的本地 Whisper。                                                                                                     |
| `--whisper_filters`                     | emoji_filter,repetition_filter | 应用于语音转文字结果的过滤器，用 "," 分隔。我们提供 emoji_filter、repetition_filter 和 japanese_stream_filter。                                                           |
| `--transcription_initial_prompt`        |                                | 通用的转录固定提示词/术语表。格式："提示词1, 提示词2, ..."。此文本将始终包含在传递给模型的提示词中。                                                                      |
| `--disable_transcription_context`       |                                | 设置此标志以禁用转录中的上下文（上一句）传递。                                                                                                                            |
| **翻译选项**                            |                                |                                                                                                                                                                           |
| `--gpt_model`                           | gpt-5-nano                     | OpenAI 的 GPT 模型名称，gpt-5 / gpt-5-mini / gpt-5-nano                                                                                                                   |
| `--gemini_model`                        | gemini-2.5-flash-lite          | Google 的 Gemini 模型名称，gemini-2.0-flash / gemini-2.5-flash / gemini-2.5-flash-lite                                                                                    |
| `--translation_prompt`                  |                                | 如果使用，将通过 GPT / Gemini API (根据填写的 API 密钥决定) 将结果文本翻译成目标语言。示例："Translate from Japanese to Chinese"                                          |
| `--translation_history_size`            | 0                              | 调用 GPT / Gemini API 时发送的历史消息作为上下文的数量。如果历史大小为 0，翻译将并行运行。如果历史大小 > 0，翻译将串行运行。                                              |
| `--translation_timeout`                 | 10                             | 如果 GPT / Gemini 当一句话翻译超过此秒数，这句话将被放弃。                                                                                                                |
| `--gpt_base_url`                        | https://api.openai.com/v1      | 自定义 GPT 的 API 端点。                                                                                                                                                  |
| `--gemini_base_url`                     |                                | 自定义 Gemini 的 API 端点。                                                                                                                                               |
| `--processing_proxy`                    |                                | 为 Whisper/GPT API 使用指定的 HTTP/HTTPS/SOCKS 代理 (Gemini 目前不支持在程序内指定代理)，例如 http://127.0.0.1:7890。                                                     |
| `--use_json_result`                     |                                | 针对某些本地部署的模型，在 LLM 翻译中使用 JSON 结果。                                                                                                                     |
| `--retry_if_translation_fails`          |                                | 当翻译超时/失败时重试。用于离线生成字幕。                                                                                                                                 |
| **输出选项**                            |                                |                                                                                                                                                                           |
| `--output_timestamps`                   |                                | 输出文本时，同时输出文本的时间戳。                                                                                                                                        |
| `--hide_transcribe_result`              |                                | 隐藏 Whisper 转录的结果。                                                                                                                                                 |
| `--output_proxy`                        |                                | 为 Cqhttp/Discord/Telegram 使用指定的 HTTP/HTTPS/SOCKS 代理，例如 http://127.0.0.1:7890。                                                                                 |
| `--output_file_path`                    |                                | 如果使用，将把结果文本保存到此路径。                                                                                                                                      |
| `--cqhttp_url`                          |                                | 如果使用，将把结果文本发送到 cqhttp 服务器。                                                                                                                              |
| `--cqhttp_token` code_snippet_pre       |                                | cqhttp 的 Token，如果服务器端未设置，则无需填写。                                                                                                                         |
| `--discord_webhook_url`                 |                                | 如果使用，将把结果文本发送到 Discord 频道。                                                                                                                               |
| `--telegram_token`                      |                                | Telegram 机器人的 Token。                                                                                                                                                 |
| `--telegram_chat_id`                    |                                | 如果使用，将把结果文本发送到此 Telegram 聊天。需要与 \"--telegram_token\" 配合使用。                                                                                      |

## 联系我

Telegram: [@ionic_bond](https://t.me/ionic_bond)

## 捐赠

[PayPal Donate](https://www.paypal.com/donate/?hosted_button_id=D5DRBK9BL6DUA) 或 [PayPal](https://paypal.me/ionicbond3)
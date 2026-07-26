# stream-translator-gpt

[![PyPI version](https://badge.fury.io/py/stream-translator-gpt.svg)](https://badge.fury.io/py/stream-translator-gpt) [![Python Versions](https://img.shields.io/pypi/pyversions/stream-translator-gpt.svg)](https://pypi.org/project/stream-translator-gpt/) [![Downloads](https://static.pepy.tech/badge/stream-translator-gpt)](https://pepy.tech/project/stream-translator-gpt) [![License](https://img.shields.io/github/license/ionic-bond/stream-translator-gpt.svg)](https://github.com/ionic-bond/stream-translator-gpt/blob/main/LICENSE) [![Gradio](https://img.shields.io/badge/WebUI-Gradio-orange)](https://gradio.app)

[English](./README.md) | 中文 | [日本語](./README_JP.md)

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
        ce("`**HuggingFace ASR**`")
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

在本地使用 [**Whisper**](https://github.com/openai/whisper) / [**Faster-Whisper**](https://github.com/SYSTRAN/faster-whisper) / [**Simul Streaming**](https://github.com/ufal/SimulStreaming) / [**HuggingFace ASR**](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) 或远程调用 [**OpenAI Transcription API**](https://platform.openai.com/docs/guides/speech-to-text) 进行转录。

使用 OpenAI 的 [**GPT API**](https://platform.openai.com/docs/overview) / Google 的 [**Gemini API**](https://ai.google.dev/gemini-api/docs) 进行翻译。

最后，结果可以打印到终端、保存到文件，或通过社交媒体机器人发送到群组。

## 准备工作

1. **Python** >= 3.10
2. **FFmpeg**（如果您的系统已安装 FFmpeg 可跳过此步）：
   - Windows: `winget install ffmpeg`
   - Linux (Debian/Ubuntu): `sudo apt install ffmpeg`
3. [**在您的系统上安装 CUDA**](https://developer.nvidia.com/cuda-downloads)。
4. 如果您想使用 **Faster-Whisper**，[**请将 cuDNN 安装到您的 CUDA 目录**](https://developer.nvidia.com/cudnn-downloads)。
5. [**为您的 Python 安装 PyTorch (CUDA 版本)**](https://pytorch.org/get-started/locally/)。
6. 如果您想使用 **Gemini API** 进行翻译，[**请创建一个 Google API 密钥**](https://aistudio.google.com/app/apikey)。
7. 如果您想使用 **OpenAI Transcription API** 进行语音转文字或使用 **GPT API** 进行翻译，[**请创建一个 OpenAI API 密钥**](https://platform.openai.com/api-keys)。

## 安装

### WebUI

```
pip install stream-translator-gpt[webui] -U
```

### 命令行

```
pip install stream-translator-gpt -U
```

## 使用方法

Colab上的命令 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/stream_translator.ipynb) 即为推荐的使用方式，以下是一些其他常用选项。

- 转录直播流 (默认使用 **Whisper**):

    ```stream-translator-gpt {网址} --language {输入语言}```

- 使用 **Faster-Whisper** 进行转录:

    ```stream-translator-gpt {网址} --language {输入语言} --use-faster-whisper```

- 使用 **SimulStreaming** 进行转录:

    ```stream-translator-gpt {网址} --language {输入语言} --use-simul-streaming```

- 使用以 **Faster-Whisper** 作为编码器的 **SimulStreaming** 进行转录:

    ```stream-translator-gpt {网址} --language {输入语言} --use-simul-streaming --use-faster-whisper```

- 使用 **OpenAI Transcription API** 进行转录:

    ```stream-translator-gpt {网址} --language {输入语言} --use-openai-transcription-api --openai-api-key {您的 OpenAI 密钥}```

- 使用 **HuggingFace ASR** 模型进行转录（需要先执行 `pip install stream-translator-gpt[hf_asr]`）：

    ```stream-translator-gpt {网址} --model {hf_model_name} --use-hf-asr```

    仅支持在 Hugging Face Hub 上 `pipeline_tag` 为 `automatic-speech-recognition` 的模型。

- 使用 **Gemini** 翻译成其他语言:

    ```stream-translator-gpt {网址} --language ja --translation-prompt "翻译以下日语为中文，只输出译文，不要输出原文，在一行内输出" --google-api-key {您的 Google 密钥}```

- 使用 **GPT** 翻译成其他语言:

    ```stream-translator-gpt {网址} --language ja --translation-prompt "翻译以下日语为中文，只输出译文，不要输出原文，在一行内输出" --openai-api-key {您的 OpenAI 密钥}```

- 同时使用 **OpenAI Transcription API** 和 **Gemini**:

    ```stream-translator-gpt {网址} --language ja --use-openai-transcription-api --openai-api-key {您的 OpenAI 密钥} --translation-prompt "翻译以下日语为中文，只输出译文，不要输出原文，在一行内输出" --google-api-key {您的 Google 密钥}```

- 使用本地视频/音频文件作为输入:

    ```stream-translator-gpt {文件路径} --language {输入语言}```

- 录制系统声音作为输入:

    ```stream-translator-gpt device --language {输入语言}```

- 录制麦克风作为输入:

    ```stream-translator-gpt device --language {输入语言} --mic```

- 发送结果到 Discord:

    ```stream-translator-gpt {网址} --language {输入语言} --discord-webhook-url {您的_discord_webhook_网址}```

- 发送结果到 Telegram:

    ```stream-translator-gpt {网址} --language {输入语言} --telegram-token {您的 Telegram 令牌} --telegram-chat-id {您的 Telegram 聊天 id}```

- 发送结果到 Cqhttp:

    ```stream-translator-gpt {网址} --language {输入语言} --cqhttp-url {您的 cqhttp 地址} --cqhttp-token {您的 cqhttp 令牌}```

- 保存结果到 .srt 字幕文件:

    ```stream-translator-gpt {网址} --language ja --translation-prompt "翻译以下日语为中文，只输出译文，不要输出原文，在一行内输出" --google-api-key {您的 Google 密钥} --no-show-transcribe-result --retry-if-translation-fails --output-timestamps --output-file-path ./result.srt```

### 所有选项

```stream-translator-gpt URL [OPTIONS]```

> [!NOTE]
> 所有选项的横杠和下划线写法等价：`--openai-api-key` 与 `--openai_api_key` 相同。
> 每个布尔选项都有对应的 `--no-*` 反向形式用于关闭（例如默认开启的 `--dynamic-vad-threshold` 可以用 `--no-dynamic-vad-threshold` 关闭）。

| 选项                                    | 默认值                         | 描述                                                                                                                                                                      |
| :-------------------------------------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `URL`                                   |                                | 直播流的 URL。如果填入本地文件路径，则会将其用作输入。如果填入 "device"，将从您的 PC 设备获取输入。                                                                       |
| **通用选项**                            |
| `--openai-api-key`                      |                                | 如果使用 GPT 翻译 / Whisper API，则需要 OpenAI API 密钥。如果您有多个密钥，可以用 "," 分隔，每个密钥将轮流使用。                                                          |
| `--google-api-key`                      |                                | 如果使用 Gemini 翻译，则需要 Google API 密钥。如果您有多个密钥，可以用 "," 分隔，每个密钥将轮流使用。                                                                     |
| `--openai-base-url`                     |                                | 自定义 OpenAI 的 API 端点 (影响 GPT 翻译和 OpenAI Whisper 转录)。                                                                                                         |
| `--google-base-url`                     |                                | 自定义 Google 的 API 端点 (影响 Gemini 翻译)。                                                                                                                            |
| `--no-verify-ssl`                       |                                | 关闭 OpenAI / Google API 和 HuggingFace 下载的 TLS 证书校验。当您的 API 端点或代理使用自签名或无效证书时使用。如果 base URL 的主机是裸 IP，则会自动关闭证书校验。         |
| `--proxy`                               |                                | 用于设置所有未特别指定的 --*-proxy 的值。也会设置 http_proxy 等环境变量。                                                                                                 |
| **输入选项**                            |                                |                                                                                                                                                                           |
| `--format`                              | ba/wa*                         | 码流格式代码，此参数将直接传递给 yt-dlp。您可以通过 `yt-dlp {url} -F` 获取可用格式代码的列表。                                                                            |
| `--list-format`                         |                                | 打印所有可用格式然后退出。                                                                                                                                                |
| `--cookies`                             |                                | 用于打开会员专属直播，此参数将直接传递给 yt-dlp。                                                                                                                         |
| `--input-proxy`                         |                                | 为 yt-dlp 使用指定的 HTTP/HTTPS/SOCKS 代理，例如 http://127.0.0.1:7890。                                                                                                  |
| `--device-index`                        |                                | 需要录制的设备的索引。如果未设置，将使用系统默认的录音设备。                                                                                                              |
| `--list-devices`                        |                                | 打印所有音频设备信息然后退出。                                                                                                                                            |
| `--device-recording-interval`           | 0.5                            | 录制间隔越短，延迟越低，但会增加 CPU 使用率。建议设置在 0.1 和 1.0 之间。                                                                                                 |
| **音频切片选项**                        |                                |                                                                                                                                                                           |
| `--min-audio-length`                    | 0.5                            | 最小音频切片长度（秒）。                                                                                                                                                  |
| `--max-audio-length`                    | 30.0                           | 最大音频切片长度（秒）。                                                                                                                                                  |
| `--target-audio-length`                 | 5.0                            | 当启用动态无语音阈值时（默认启用），程序将尽可能按接近此长度切割音频。                                                                                                    |
| `--continuous-no-speech-threshold`      | 1.0                            | 如果在此秒数内没有语音，则进行切片。如果启用了动态无语音阈值（默认启用），实际阈值将基于此值动态调整。                                                                    |
| `--no-dynamic-no-speech-threshold` |                                | 禁用动态静音阈值（默认开启）。                                                                                                                                            |
| `--prefix-retention-length`             | 0.5                            | 切片时保留的前缀音频长度。                                                                                                                                                |
| `--vad-threshold`                       | 0.35                           | 范围 0~1。此值越高，语音判断越严格。如果启用了动态 VAD 阈值（默认启用），此阈值将根据输入语音的 VAD 结果动态调整。                                                        |
| `--no-dynamic-vad-threshold`       |                                | 禁用动态 VAD 阈值（默认开启）。                                                                                                                                           |
| **转录选项**                            |                                |                                                                                                                                                                           |
| `--model`                               | turbo                          | 选择 Whisper/Faster-Whisper/Simul Streaming 模型大小。可用模型请参见 [此处](https://github.com/openai/whisper#available-models-and-languages)。                           |
| `--language`                            | auto                           | 直播流中的语言。可用语言请参见 [此处](https://github.com/openai/whisper#available-models-and-languages)。                                                                 |
| `--use-faster-whisper`                  |                                | 设置此标志以使用 Faster-Whisper 进行语音转文字，而不是原始的 OpenAI Whisper。如果与 --use-simul-streaming 一起使用，将使用以 Faster-Whisper 作为编码器的 SimulStreaming。 |
| `--use-simul-streaming`                 |                                | 设置此标志以使用 SimulStreaming 进行语音转文字，而不是原始的 OpenAI Whisper。如果与 --use-faster-whisper 一起使用，将使用以 Faster-Whisper 作为编码器的 SimulStreaming。  |
| `--use-openai-transcription-api`        |                                | 设置此标志以使用 OpenAI transcription API，而不是原始的本地 Whisper。                                                                                                     |
| `--use-hf-asr`                          |                                | 设置此标志以使用 HuggingFace ASR 模型。通过 `--model` 指定模型 ID。需要先执行 `pip install stream-translator-gpt[hf_asr]`。                                               |
| `--transcription-filters`               | emoji_filter,repetition_filter | 应用于语音转文字结果的过滤器，用 "," 分隔。提供 emoji_filter 和 repetition_filter。                                                        |
| `--no-language-based-filter`        |                                | 禁用根据 ASR 语言自动挂载的语言过滤器（默认开启）。目前我们提供英文、中文和日文语言的过滤器。       |
| `--transcription-initial-prompt`        |                                | 通用的转录固定提示词/术语表。格式："提示词1, 提示词2, ..."。此文本将始终包含在传递给模型的提示词中。                                                                      |
| `--no-transcription-context`       |                                | 禁用转录中的上下文（上一句）传递（默认开启）。                                                                                                                            |
| **翻译选项**                            |
| `--gpt-model`                           | gpt-5.4-nano                   | OpenAI 的 GPT 模型名称，gpt-5.4 / gpt-5.4-mini / gpt-5.4-nano / gpt-5.5 / gpt-5.6-luna                                                                                |
| `--gemini-model`                        | gemini-3.5-flash-lite          | Google 的 Gemini 模型名称，gemini-3-flash-preview / gemini-3.1-flash-lite / gemini-3.5-flash / gemini-3.5-flash-lite / gemini-3.6-flash |
| `--translation-prompt`                  |                                | 如果使用，将通过 GPT / Gemini API (根据填写的 API 密钥决定) 将结果文本翻译成目标语言。示例："Translate from Japanese to Chinese"                                          |
| `--translation-history-size`            | 0                              | 调用 LLM API 时作为上下文发送的先前转录数量。建议对较弱的模型禁用上下文（设置为 0）。                                                                                     |
| `--translation-timeout`                 | 10                             | 如果 GPT / Gemini 当一句话翻译超过此秒数，这句话将被放弃。                                                                                                                |
| `--use-json-result`                     |                                | 针对某些本地部署的模型，在 LLM 翻译中使用 JSON 结果。                                                                                                                     |
| `--retry-if-translation-fails`          |                                | 当翻译超时/失败时重试。用于离线生成字幕。                                                                                                                                 |
| `--temperature`                         |                                | GPT/Gemini 参数。控制输出随机性，值越高结果越多样。                                                                                                                       |
| `--top-p`                               |                                | GPT/Gemini 参数。核采样阈值，仅考虑累计概率超过该值的 token。                                                                                                             |
| `--top-k`                               |                                | Gemini 参数。将 token 选择限制为概率最高的 K 个候选项。                                                                                                                   |
| `--prompt-cache-key`                    |                                | GPT 参数。设置后启用 API 端的提示词缓存优化。                                                                                                                             |
| `--reasoning-effort`                    |                                | GPT 参数。控制推理模型的推理深度。可选值：none / minimal / low / medium / high / xhigh。                                                                                  |
| `--verbosity`                           |                                | GPT 参数。控制回复的详细程度。可选值：auto / short / concise / detailed。                                                                                                 |
| `--service-tier`                        |                                | GPT 参数。指定处理优先级。可选值：auto / default / flex / priority。                                                                                                      |
| `--debug-mode`                          |                                | 启用调试模式。打印发送给 LLM 的消息以及每次翻译调用后的使用信息。                                                                                                         |
| `--processing-proxy`                    |                                | 为 Whisper/GPT API 使用指定的 HTTP/HTTPS/SOCKS 代理 (Gemini 目前不支持在程序内指定代理)，例如 http://127.0.0.1:7890。                                                     |
| **输出选项**                            |
| `--output-timestamps`                   |                                | 输出文本时，同时输出文本的时间戳。                                                                                                                                        |
| `--no-show-transcribe-result`              |                                | 隐藏 Whisper 转录的结果（默认显示）。                                                                                                                                                 |
| `--output-file-path`                    |                                | 如果使用，将把结果文本保存到此路径。                                                                                                                                      |
| `--cqhttp-url`                          |                                | 如果使用，将把结果文本发送到 cqhttp 服务器。                                                                                                                              |
| `--cqhttp-token`                        |                                | cqhttp 的 Token，如果服务器端未设置，则无需填写。                                                                                                                         |
| `--discord-webhook-url`                 |                                | 如果使用，将把结果文本发送到 Discord 频道。                                                                                                                               |
| `--telegram-token`                      |                                | Telegram 机器人的 Token。                                                                                                                                                 |
| `--telegram-chat-id`                    |                                | 如果使用，将把结果文本发送到此 Telegram 聊天。需要与 \"--telegram-token\" 配合使用。                                                                                      |
| `--output-proxy`                        |                                | 为 Cqhttp/Discord/Telegram 使用指定的 HTTP/HTTPS/SOCKS 代理，例如 http://127.0.0.1:7890。                                                                                 |

## 联系我

Telegram: [@ionic_bond](https://t.me/ionic_bond)

## 捐赠

[PayPal Donate](https://www.paypal.com/donate/?hosted_button_id=D5DRBK9BL6DUA) 或 [PayPal](https://paypal.me/ionicbond3)

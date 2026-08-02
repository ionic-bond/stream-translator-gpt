import dataclasses
import os
import platform
import queue
import shutil
import signal
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import tyro

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "stream_translator_gpt"

from .common import ClientPool, start_daemon_thread, is_url, WARNING, ERROR, INFO
from .audio_getter import StreamAudioGetter, LocalFileAudioGetter, DeviceAudioGetter
from .audio_slicer import AudioSlicer
from .audio_transcriber import OpenaiWhisper, FasterWhisper, SimulStreaming, RemoteOpenaiTranscriber, HFTranscriber
from .llm_translator import GPTTranslator, GeminiTranslator
from .result_exporter import ResultExporter
from . import __version__


@dataclasses.dataclass
class Config:
    url: Annotated[tyro.conf.Positional[str], tyro.conf.arg(metavar='URL')]
    """The URL of the stream. If a local file path is filled in, it will be used as input. If fill in "device", the
    input will be obtained from your PC device."""

    openai_api_key: str | None = None
    """OpenAI API key if using GPT translation / Whisper API. If you have multiple keys, you can separate them with ","
    and each key will be used in turn."""

    google_api_key: str | None = None
    """Google API key if using Gemini translation. If you have multiple keys, you can separate them with "," and each
    key will be used in turn."""

    openai_base_url: str | None = None
    """Customize the API endpoint of OpenAI (Affects GPT translation & OpenAI Transcription)."""

    google_base_url: str | None = None
    """Customize the API endpoint of Google (Affects Gemini translation)."""

    verify_ssl: bool = True
    """TLS certificate verification for OpenAI / Google API and HuggingFace downloads. Use --no-verify-ssl when your
    API endpoint or proxy has a self-signed or invalid certificate. If the base URL host is a bare IP, verification is
    disabled automatically."""

    proxy: str | None = None
    """Used to set the proxy for all --*-proxy flags if they are not specifically set. Also sets http_proxy environment
    variables."""

    format: str = 'ba/wa*'
    """Stream format code, this parameter will be passed directly to yt-dlp. You can get the list of available format
    codes by "yt-dlp {url} -F"."""

    list_format: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False
    """Print all available formats then exit."""

    cookies: str | None = None
    """Used to open member-only stream, this parameter will be passed directly to yt-dlp."""

    input_proxy: str | None = None
    """Use the specified HTTP/HTTPS/SOCKS proxy for yt-dlp, e.g. http://127.0.0.1:7890."""

    device_index: int | None = None
    """The index of the device that needs to be recorded. If not set, the system default recording device will be
    used."""

    device_recording_interval: float = 0.5
    """The shorter the recording interval, the lower the latency, but it will increase CPU usage. It is recommended to
    set it between 0.1 and 1.0."""

    list_devices: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False
    """Print all audio devices info then exit."""

    mic: bool = False
    """Use microphone instead of system audio (loopback)."""

    min_audio_length: float = 0.5
    """Minimum slice audio length in seconds."""

    max_audio_length: float = 30.0
    """Maximum slice audio length in seconds."""

    target_audio_length: float = 5.0
    """When dynamic no speech threshold is enabled (enabled by default), the program will slice the audio as close to
    this length as possible."""

    continuous_no_speech_threshold: float = 1.0
    """Slice if there is no speech during this number of seconds. If the dynamic no speech threshold is enabled
    (enabled by default), the actual threshold will be dynamically adjusted based on this value."""

    dynamic_no_speech_threshold: bool = True
    """Dynamically adjust the no speech threshold based on --continuous-no-speech-threshold. Disable with
    --no-dynamic-no-speech-threshold."""

    prefix_retention_length: float = 0.5
    """The length of the retention prefix audio during slicing."""

    vad_threshold: float = 0.35
    """Range 0~1. The higher this value, the stricter the speech judgment. If dynamic VAD threshold is enabled (enabled
    by default), this threshold will be adjusted dynamically based on this value."""

    dynamic_vad_threshold: bool = True
    """Dynamically adjust the VAD threshold based on --vad-threshold. Disable with --no-dynamic-vad-threshold."""

    model: str = 'turbo'
    """Select Whisper/Faster-Whisper/SimulStreaming model size. See
    https://github.com/openai/whisper#available-models-and-languages for available models."""

    language: str | None = 'auto'
    """Language spoken in the stream. Default option is to auto detect the spoken language. See
    https://github.com/openai/whisper#available-models-and-languages for available languages."""

    use_faster_whisper: bool = False
    """Use Faster-Whisper instead of Whisper. If used with --use-simul-streaming, SimulStreaming with Faster-Whisper
    as the encoder will be used."""

    use_simul_streaming: bool = False
    """Use SimulStreaming instead of Whisper. If used with --use-faster-whisper, SimulStreaming with Faster-Whisper as
    the encoder will be used."""

    use_openai_transcription_api: bool = False
    """Use OpenAI Transcription API instead of the original local Whisper."""

    openai_transcription_model: str = 'gpt-transcribe'
    """OpenAI's transcription model name, gpt-transcribe / whisper-1 / gpt-4o-mini-transcribe /
    gpt-4o-transcribe."""

    use_hf_asr: bool = False
    """Use a HuggingFace ASR model (via transformers pipeline) specified by --model."""

    transcription_filters: str = 'emoji_filter,repetition_filter'
    """Filters apply to transcription results, separated by ",". We provide emoji_filter and repetition_filter."""

    language_based_filter: bool = True
    """Language-based transcription filters (e.g. english_filter, chinese_filter, japanese_filter) selected by ASR
    language. Disable with --no-language-based-filter."""

    transcription_keywords: str | None = None
    """Comma-separated transcription keywords. OpenAI gpt-transcribe sends them as keywords; other Whisper backends
    use them as an initial prompt. SimulStreaming uses them only during initialization; HuggingFace ASR ignores them."""

    transcription_context: bool = False
    """Pass the previous transcription result as context. Only one result is retained. SimulStreaming and HuggingFace
    ASR do not support text context propagation. Enable with --transcription-context, disabled by default."""

    gpt_model: str = 'gpt-5.6-luna'
    """OpenAI's GPT model name, gpt-5.4-nano / gpt-5.4-mini / gpt-5.6-luna / gpt-5.6-terra."""

    gemini_model: str = 'gemini-3.5-flash-lite'
    """Google's Gemini model name, gemini-3-flash-preview / gemini-3.1-flash-lite / gemini-3.5-flash /
    gemini-3.5-flash-lite / gemini-3.6-flash."""

    translation_prompt: str | None = None
    """If set, will translate result text to target language via GPT / Gemini API. Example: "Translate from Japanese
    to Chinese". Adding context (who the streamer is, what the stream is about) improves translation quality."""

    translation_history_size: int = 3
    """The number of previous transcripts sent as context when calling the LLM API. It is recommended to disable
    context (set to 0) for weaker models."""

    translation_timeout: int = 10
    """If the GPT / Gemini translation exceeds this number of seconds, the translation will be discarded."""

    use_json_result: bool = False
    """Using JSON result in LLM translation for some locally deployed models."""

    retry_if_translation_fails: bool = False
    """Retry when translation times out/fails. Used to generate subtitles offline."""

    temperature: float | None = None
    """GPT/Gemini parameter. Controls output randomness, higher values produce more diverse results."""

    top_p: float | None = None
    """GPT/Gemini parameter. Nucleus sampling threshold, only tokens with cumulative probability above this value are
    considered."""

    top_k: int | None = None
    """Gemini parameter. Limits token selection to the top K most probable candidates."""

    prompt_cache_key: str | None = None
    """GPT parameter. If set, enables prompt caching optimization on the API side."""

    reasoning_effort: str | None = None
    """GPT parameter. Controls reasoning depth for reasoning models. Options: none / minimal / low / medium / high /
    xhigh / max."""

    verbosity: str | None = None
    """GPT parameter. Controls the verbosity of the response. Options: low / medium / high."""

    service_tier: str | None = None
    """GPT parameter. Specifies processing priority tier. Options: auto / default / flex / priority."""

    debug_mode: bool = False
    """Enable debug mode. Print messages sent to LLM and usage info after each translation call."""

    processing_proxy: str | None = None
    """Use the specified HTTP/HTTPS/SOCKS proxy for Whisper/GPT API (Gemini currently doesn't support specifying a
    proxy within the program), e.g. http://127.0.0.1:7890."""

    output_timestamps: bool = False
    """Output the timestamp of the text when outputting the text."""

    show_transcribe_result: bool = True
    """Print / export the transcription result. Disable with --no-show-transcribe-result to only output the
    translation."""

    output_file_path: str | None = None
    """If set, will save the result text to this path."""

    cqhttp_url: str | None = None
    """If set, will send the result text to this Cqhttp server."""

    cqhttp_token: str | None = None
    """Token of cqhttp, if it is not set on the server side, it does not need to fill in."""

    discord_webhook_url: str | None = None
    """If set, will send the result text to this Discord channel."""

    telegram_token: str | None = None
    """Token of Telegram bot."""

    telegram_chat_id: int | None = None
    """If set, will send the result text to this Telegram chat. Needs to be used with --telegram-token."""

    output_proxy: str | None = None
    """Use the specified HTTP/HTTPS/SOCKS proxy for Cqhttp/Discord/Telegram, e.g. http://127.0.0.1:7890."""


def run(config: Config):
    """Run the transcription / translation pipeline until the input is exhausted."""
    ClientPool.init(openai_api_key=config.openai_api_key,
                    google_api_key=config.google_api_key,
                    proxy=config.processing_proxy,
                    openai_base_url=config.openai_base_url,
                    google_base_url=config.google_base_url,
                    verify_ssl=config.verify_ssl)

    # Init queues
    getter_to_slicer_queue = queue.SimpleQueue()
    slicer_to_transcriber_queue = queue.SimpleQueue()
    transcriber_to_translator_queue = queue.SimpleQueue()
    translator_to_exporter_queue = queue.SimpleQueue() if config.translation_prompt else transcriber_to_translator_queue

    # Init workers
    with ThreadPoolExecutor() as executor:

        def init_audio_getter():
            if config.url.lower() == 'device':
                return DeviceAudioGetter(
                    device_index=config.device_index,
                    use_mic=config.mic,
                    interval=config.device_recording_interval,
                )
            elif is_url(config.url):
                return StreamAudioGetter(
                    url=config.url,
                    format=config.format,
                    cookies=config.cookies,
                    proxy=config.input_proxy,
                )
            else:
                return LocalFileAudioGetter(file_path=config.url)

        audio_getter_future = executor.submit(init_audio_getter)
        slicer_future = executor.submit(
            AudioSlicer,
            min_audio_length=config.min_audio_length,
            max_audio_length=config.max_audio_length,
            target_audio_length=config.target_audio_length,
            continuous_no_speech_threshold=config.continuous_no_speech_threshold,
            dynamic_no_speech_threshold=config.dynamic_no_speech_threshold,
            prefix_retention_length=config.prefix_retention_length,
            vad_threshold=config.vad_threshold,
            dynamic_vad_threshold=config.dynamic_vad_threshold,
        )

        def init_transcriber():
            common_args = {
                'transcription_filters': config.transcription_filters,
                'language_based_filter': config.language_based_filter,
                'print_result': config.show_transcribe_result,
                'output_timestamps': config.output_timestamps,
                'use_history_context': config.transcription_context,
                'transcription_keywords': config.transcription_keywords,
            }
            if config.use_simul_streaming:
                return SimulStreaming(model=config.model,
                                      language=config.language,
                                      use_faster_whisper=config.use_faster_whisper,
                                      proxy=config.processing_proxy,
                                      **common_args)
            elif config.use_faster_whisper:
                return FasterWhisper(model=config.model,
                                     language=config.language,
                                     proxy=config.processing_proxy,
                                     **common_args)
            elif config.use_openai_transcription_api:
                return RemoteOpenaiTranscriber(model=config.openai_transcription_model,
                                               language=config.language,
                                               proxy=config.processing_proxy,
                                               **common_args)
            elif config.use_hf_asr:
                return HFTranscriber(model=config.model,
                                     language=config.language,
                                     proxy=config.processing_proxy,
                                     **common_args)
            else:
                return OpenaiWhisper(model=config.model, language=config.language, **common_args)

        transcriber_future = executor.submit(init_transcriber)

        def init_translator():
            if not config.translation_prompt:
                return None
            common_args = {
                'prompt': config.translation_prompt,
                'history_size': config.translation_history_size,
                'use_json_result': config.use_json_result,
                'timeout': config.translation_timeout,
                'retry_if_translation_fails': config.retry_if_translation_fails,
                'debug_mode': config.debug_mode,
            }
            if config.google_api_key:
                return GeminiTranslator(
                    model=config.gemini_model,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    **common_args,
                )
            else:
                return GPTTranslator(
                    model=config.gpt_model,
                    prompt_cache_key=config.prompt_cache_key,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    reasoning_effort=config.reasoning_effort,
                    verbosity=config.verbosity,
                    service_tier=config.service_tier,
                    **common_args,
                )

        translator_future = executor.submit(init_translator)
        exporter_future = executor.submit(
            ResultExporter,
            cqhttp_url=config.cqhttp_url,
            cqhttp_token=config.cqhttp_token,
            discord_webhook_url=config.discord_webhook_url,
            telegram_token=config.telegram_token,
            telegram_chat_id=config.telegram_chat_id,
            output_file_path=config.output_file_path,
            proxy=config.output_proxy,
            output_whisper_result=config.show_transcribe_result,
            output_timestamps=config.output_timestamps,
        )

        audio_getter = audio_getter_future.result()
        slicer = slicer_future.result()
        transcriber = transcriber_future.result()
        translator = translator_future.result()
        exporter = exporter_future.result()

    if hasattr(audio_getter, '_exit_handler'):
        signal.signal(signal.SIGINT, audio_getter._exit_handler)

    print(f'{INFO}Initialization complete, starting up...')

    # Start working
    start_daemon_thread(audio_getter.loop, output_queue=getter_to_slicer_queue)
    start_daemon_thread(
        slicer.loop,
        input_queue=getter_to_slicer_queue,
        output_queue=slicer_to_transcriber_queue,
    )
    start_daemon_thread(
        transcriber.loop,
        input_queue=slicer_to_transcriber_queue,
        output_queue=transcriber_to_translator_queue,
    )
    if translator:
        start_daemon_thread(
            translator.loop,
            input_queue=transcriber_to_translator_queue,
            output_queue=translator_to_exporter_queue,
        )
    exporter_thread = start_daemon_thread(
        exporter.loop,
        input_queue=translator_to_exporter_queue,
    )

    while exporter_thread.is_alive():
        time.sleep(1)
    print(f'{INFO}All processing completed, program exits.')


def _preprocess_deprecated_flags(argv):
    """Rewrite deprecated flags to their new form, printing a warning for each."""
    processed = []
    for arg in argv:
        if arg.replace('_', '-') == '--hide-transcribe-result':
            print(f'{WARNING}--hide_transcribe_result is deprecated and will be removed in future versions. '
                  'Please use --no-show-transcribe-result instead.')
            processed.append('--no-show-transcribe-result')
        elif arg.replace('_', '-') == '--transcription-initial-prompt':
            print(f'{WARNING}--transcription-initial-prompt is deprecated and will be removed in future versions. '
                  'Please use --transcription-keywords instead.')
            processed.append('--transcription-keywords')
        else:
            processed.append(arg)
    return processed


def _apply_overall_proxy(config: Config):
    """Propagate --proxy to the environment and to all unset --*-proxy options."""
    if not config.proxy:
        return
    os.environ['http_proxy'] = config.proxy
    os.environ['https_proxy'] = config.proxy
    os.environ['HTTP_PROXY'] = config.proxy
    os.environ['HTTPS_PROXY'] = config.proxy
    if config.input_proxy is None:
        config.input_proxy = config.proxy
    if config.processing_proxy is None:
        config.processing_proxy = config.proxy
    if config.output_proxy is None:
        config.output_proxy = config.proxy


def _print_audio_devices():
    if platform.system() == 'Windows':
        import pyaudiowpatch as pa
    else:
        try:
            import pyaudio as pa
        except ImportError:
            print("PyAudio is not installed. Unable to list devices.")
            print("Debian/Ubuntu/Colab: apt install portaudio19-dev && pip install pyaudio")
            sys.exit(1)

    pyaudio = pa.PyAudio()

    print("Available audio devices:")
    for i in range(pyaudio.get_device_count()):
        dev = pyaudio.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            print(f"{dev['index']}: {dev['name']}")

    if platform.system() == 'Windows':
        print("\nLoopback devices (for system audio):")
        for loopback in pyaudio.get_loopback_device_info_generator():
            print(f"{loopback['index']}: {loopback['name']}")
    pyaudio.terminate()


def _print_stream_formats(config: Config):
    cmd = [sys.executable, '-m', 'yt_dlp', config.url, '-F']
    if config.cookies:
        cmd.extend(['--cookies', config.cookies])
    if config.input_proxy:
        cmd.extend(['--proxy', config.input_proxy])
    subprocess.run(cmd)


def _check_ffmpeg(config: Config):
    if config.url.lower() != 'device' and not shutil.which('ffmpeg'):
        if platform.system() == 'Windows':
            print(f'{ERROR}ffmpeg not found. Please install it with: winget install ffmpeg')
        else:
            print(f'{ERROR}ffmpeg not found. Please install it with: sudo apt install ffmpeg')
        sys.exit(1)


def _validate_and_normalize(config: Config):
    """Check option combinations and normalize values, exiting with an error message on invalid input."""
    if config.model.endswith('.en'):
        if config.model == 'large.en':
            print(
                f'{ERROR}English model does not have large model, please choose from {{tiny.en, small.en, medium.en}}')
            sys.exit(1)
        if config.language != 'English' and config.language != 'en':
            if config.language == 'auto':
                print(f'{WARNING}Using .en model, setting language from auto to English')
                config.language = 'en'
            else:
                print(
                    f'{ERROR}English model cannot be used to detect non english language, please choose a non .en model'
                )
                sys.exit(1)

    transcription_encoder_flag_num = 0
    transcription_decoder_flag_num = 0
    if config.use_faster_whisper:
        transcription_encoder_flag_num += 1
    if config.use_simul_streaming:
        transcription_decoder_flag_num += 1
    if config.use_openai_transcription_api:
        transcription_encoder_flag_num += 1
        transcription_decoder_flag_num += 1
    if config.use_hf_asr:
        transcription_encoder_flag_num += 1
        transcription_decoder_flag_num += 1
    if transcription_encoder_flag_num > 1:
        print(f'{ERROR}Cannot use Faster Whisper, OpenAI Transcription API or HuggingFace ASR at the same time')
        sys.exit(1)
    if transcription_decoder_flag_num > 1:
        print(f'{ERROR}Cannot use SimulStreaming, OpenAI Transcription API or HuggingFace ASR at the same time')
        sys.exit(1)

    if config.use_openai_transcription_api and not config.openai_api_key:
        print(f'{ERROR}Please fill in the OpenAI API key when enabling OpenAI Transcription API')
        sys.exit(1)

    if config.translation_prompt and not (config.openai_api_key or config.google_api_key):
        print(f'{ERROR}Please fill in the OpenAI / Google API key when enabling LLM translation')
        sys.exit(1)

    if config.language == 'auto':
        config.language = None

    if config.output_file_path:
        if os.path.splitext(config.output_file_path)[1].lower() == '.srt' and not config.output_timestamps:
            print(f'{WARNING}Output timestamps are required for .srt files, enabling them automatically.')
            config.output_timestamps = True
        output_dir = os.path.dirname(os.path.abspath(config.output_file_path))
        if not os.path.isdir(output_dir):
            print(f'{ERROR}Output directory does not exist: {output_dir}')
            sys.exit(1)


def cli():
    print(f'{INFO}Version: {__version__}')
    config = tyro.cli(Config, args=_preprocess_deprecated_flags(sys.argv[1:]), prog='stream-translator-gpt')

    _apply_overall_proxy(config)

    if config.list_devices:
        _print_audio_devices()
        return
    if config.list_format:
        _print_stream_formats(config)
        return

    _check_ffmpeg(config)
    _validate_and_normalize(config)
    run(config)


if __name__ == '__main__':
    cli()

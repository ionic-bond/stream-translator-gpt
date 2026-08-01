import os
import io
import queue
import re
from abc import abstractmethod
from scipy.io.wavfile import write as write_audio

import numpy as np

from . import filters
from .common import TranslationTask, SAMPLE_RATE, LoopWorkerBase, sec2str, ClientPool, INFO, WARNING
from .simul_streaming.simul_whisper.whisper.utils import compression_ratio


class AudioTranscriber(LoopWorkerBase):

    def __init__(self, language: str, transcription_filters: str, language_based_filter: bool, print_result: bool,
                 output_timestamps: bool, transcription_context: bool, transcription_initial_prompt: str):
        self.language = language
        self.transcription_filters = transcription_filters
        self.language_based_filter = language_based_filter
        self.print_result = print_result
        self.output_timestamps = output_timestamps
        self.transcription_context = transcription_context
        self.transcription_initial_prompt = transcription_initial_prompt

        self.constant_prompt = re.sub(r',\s*', ', ',
                                      transcription_initial_prompt) if transcription_initial_prompt else ""
        if self.constant_prompt and not self.constant_prompt.strip().endswith(','):
            self.constant_prompt += ','

        self.filter_chain = self._build_filter_chain()

    @abstractmethod
    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        """Returns (text, tokens). tokens can be None if not available."""
        pass

    def reset_context(self):
        """Override in subclass to reset model context when repetition is detected."""
        pass

    def _build_filter_chain(self) -> list:
        chain = []
        if self.transcription_filters:
            for filter_name in self.transcription_filters.split(','):
                filter_name = filter_name.strip()
                if not filter_name:
                    continue
                filter_func = getattr(filters, filter_name, None)
                if not filter_func:
                    print(f'{WARNING}Unknown filter "{filter_name}", skipping.')
                    continue
                if filter_func not in chain:
                    chain.append(filter_func)

        if self.language_based_filter:
            for lf in filters.get_language_filters(self.language):
                if lf not in chain:
                    chain.append(lf)

        return chain

    def filter_text(self, text: str) -> str:
        for f in self.filter_chain:
            text = f(text)
        return text

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_queue: queue.SimpleQueue[TranslationTask]):
        previous_text = ""

        while True:
            task = input_queue.get()
            if task is None:
                output_queue.put(None)
                break

            dynamic_context = filters.symbol_filter(previous_text) if self.transcription_context else ""

            if self.constant_prompt:
                limit = 500 - len(self.constant_prompt) - 1
                if len(dynamic_context) > limit:
                    if limit > 0:
                        dynamic_context = dynamic_context[-limit:]
                    else:
                        dynamic_context = ""

            initial_prompt = f"{self.constant_prompt} {dynamic_context}".strip()
            if not initial_prompt:
                initial_prompt = None

            text, tokens = self.transcribe(task.audio, initial_prompt=initial_prompt)

            if self.constant_prompt and text.strip().rstrip(',') == self.constant_prompt.strip().rstrip(','):
                text = ""

            # Repetition detection: reset context if compression ratio too high OR token diversity too low
            is_repetitive = False
            if text:
                zlib_ratio = compression_ratio(text)
                unique_ratio = len(set(tokens)) / len(tokens) if tokens else 1.0

                if zlib_ratio > 1.5 or unique_ratio < 0.6:
                    self.reset_context()
                    is_repetitive = True

            task.transcript = self.filter_text(text).strip()
            if not task.transcript:
                continue
            previous_text = "" if is_repetitive else task.transcript
            if self.print_result:
                if self.output_timestamps:
                    timestamp_text = f'{sec2str(task.time_range[0])} --> {sec2str(task.time_range[1])}'
                    print(timestamp_text + ' ' + task.transcript)
                else:
                    print(task.transcript)
            output_queue.put(task)


class OpenaiWhisper(AudioTranscriber):

    def __init__(self, model: str, language: str, **kwargs) -> None:
        super().__init__(language=language, **kwargs)
        import whisper

        print(f'{INFO}Loading Whisper model: {model}')
        self.model = whisper.load_model(model)

    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        result = self.model.transcribe(audio,
                                       without_timestamps=True,
                                       language=self.language,
                                       initial_prompt=initial_prompt)
        text = result.get('text', '')
        tokens = []
        for segment in result.get('segments', []):
            tokens.extend(segment.get('tokens', []))
        return text, tokens if tokens else None


def _apply_hf_proxy(proxy: str):
    try:
        import huggingface_hub
        session = huggingface_hub.utils.get_session()
        session.proxies = {'http': proxy, 'https': proxy}
        session.verify = ClientPool.verify_ssl
    except Exception:
        pass


class FasterWhisper(AudioTranscriber):

    def __init__(self, model: str, language: str, proxy: str, **kwargs) -> None:
        super().__init__(language=language, **kwargs)
        from faster_whisper import WhisperModel

        if proxy:
            _apply_hf_proxy(proxy)
        print(f'{INFO}Loading Faster-Whisper model: {model}')
        self.model = WhisperModel(model, device='auto', compute_type='auto')

    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        segments, info = self.model.transcribe(audio, language=self.language, initial_prompt=initial_prompt)
        text = ''
        tokens = []
        for segment in segments:
            text += segment.text
            tokens.extend(getattr(segment, 'tokens', None) or [])
        return text, tokens if tokens else None


class SimulStreaming(AudioTranscriber):

    def __init__(self, model: str, language: str, use_faster_whisper: bool, proxy: str, **kwargs) -> None:
        super().__init__(language=language, **kwargs)
        from .simul_streaming.simulstreaming_whisper import SimulWhisperASR, SimulWhisperOnline

        fw_encoder = None
        if use_faster_whisper:
            print(f'{INFO}Loading Faster-Whisper as encoder for SimulStreaming: {model}')
            from faster_whisper import WhisperModel
            if proxy:
                _apply_hf_proxy(proxy)
            fw_encoder = WhisperModel(model, device='auto', compute_type='auto')

        print(f'{INFO}Loading SimulStreaming model: {model}')
        simulstreaming_params = {
            "language": language,
            "model": model,
            "cif_ckpt_path": None,
            "frame_threshold": 25,
            "audio_max_len": 10.0,
            "audio_min_len": 0.0,
            "segment_length": 0.5,
            "task": "transcribe",
            "beams": 1,
            "decoder_type": "greedy",
            "never_fire": False,
            "init_prompt": self.constant_prompt,
            "static_init_prompt": None,
            "max_context_tokens": 50,
            "logdir": None,
            "fw_encoder": fw_encoder,
        }
        asr = SimulWhisperASR(**simulstreaming_params)
        self.asr_online = SimulWhisperOnline(asr)
        self.asr_online.init()

    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        self.asr_online.insert_audio_chunk(audio)
        result = self.asr_online.process_iter(is_last=True)
        return result.get('text', ''), result.get('tokens', None)

    def reset_context(self):
        self.asr_online.model.refresh_segment(complete=True)
        self.asr_online.unicode_buffer = []


class RemoteOpenaiTranscriber(AudioTranscriber):
    # https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=python

    def __init__(self, model: str, language: str, proxy: str, **kwargs) -> None:
        super().__init__(language=language, **kwargs)
        print(f'{INFO}Using {model} API as transcription engine.')
        self.model = model

    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        # Create an in-memory buffer
        audio_buffer = io.BytesIO()
        audio_buffer.name = 'audio.wav'
        write_audio(audio_buffer, SAMPLE_RATE, audio)
        audio_buffer.seek(0)

        call_args = {
            'model': self.model,
            'file': audio_buffer,
            'language': self.language,
        }
        if initial_prompt:
            call_args['prompt'] = initial_prompt

        client = ClientPool.get_openai_client()
        result = client.audio.transcriptions.create(**call_args).text
        return result, None


class HFTranscriber(AudioTranscriber):

    def __init__(self, model: str, language: str, proxy: str, **kwargs) -> None:
        super().__init__(language=language, **kwargs)
        from transformers import pipeline

        if proxy:
            _apply_hf_proxy(proxy)

        if not os.path.exists(model):
            try:
                from huggingface_hub import model_info
                info = model_info(model)
                tag = info.pipeline_tag
                if tag and tag != 'automatic-speech-recognition':
                    raise ValueError(
                        f'Model "{model}" has pipeline_tag="{tag}", not "automatic-speech-recognition". '
                        f'It is not compatible with --use_hf_asr. '
                        f'Please choose a model with pipeline_tag="automatic-speech-recognition" on HuggingFace Hub.')
            except ImportError:
                pass

        print(f'{INFO}Loading HuggingFace ASR model: {model}')
        self.pipe = pipeline('automatic-speech-recognition', model=model, device_map='auto')

    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        generate_kwargs = {}
        # Legacy Whisper configs reject the language argument.
        if self.language and hasattr(getattr(self.pipe, 'generation_config', None), 'lang_to_id'):
            generate_kwargs['language'] = self.language
        result = self.pipe(
            {
                'array': audio,
                'sampling_rate': SAMPLE_RATE
            },
            generate_kwargs=generate_kwargs,
        )
        return result['text'], None

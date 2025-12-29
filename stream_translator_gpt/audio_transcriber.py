import os
import io
import queue
import re
from abc import abstractmethod
from scipy.io.wavfile import write as write_audio

import numpy as np

from . import filters
from .common import TranslationTask, SAMPLE_RATE, LoopWorkerBase, sec2str, ApiKeyPool, INFO
from .simul_streaming.simul_whisper.whisper.utils import compression_ratio


def _filter_text(text: str, whisper_filters: str):
    filter_name_list = whisper_filters.split(',')
    for filter_name in filter_name_list:
        filter = getattr(filters, filter_name)
        if not filter:
            raise Exception('Unknown filter: %s' % filter_name)
        text = filter(text)
    return text


class AudioTranscriber(LoopWorkerBase):

    def __init__(self, whisper_filters: str, print_result: bool, output_timestamps: bool,
                 disable_transcription_context: bool, transcription_initial_prompt: str):
        self.whisper_filters = whisper_filters
        self.print_result = print_result
        self.output_timestamps = output_timestamps
        self.disable_transcription_context = disable_transcription_context
        self.transcription_initial_prompt = transcription_initial_prompt

        self.constant_prompt = re.sub(r',\s*', ', ',
                                      transcription_initial_prompt) if transcription_initial_prompt else ""
        if self.constant_prompt and not self.constant_prompt.strip().endswith(','):
            self.constant_prompt += ','

    @abstractmethod
    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        """Returns (text, tokens). tokens can be None if not available."""
        pass

    def reset_context(self):
        """Override in subclass to reset model context when repetition is detected."""
        pass

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_queue: queue.SimpleQueue[TranslationTask]):
        previous_text = ""

        while True:
            task = input_queue.get()
            if task is None:
                output_queue.put(None)
                break

            dynamic_context = previous_text if not self.disable_transcription_context else ""

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
            if len(text) > 10:
                zlib_ratio = compression_ratio(text)
                unique_ratio = len(set(tokens)) / len(tokens) if tokens else 1.0

                if zlib_ratio > 2.0 or unique_ratio < 0.4:
                    self.reset_context()
                    is_repetitive = True

            task.transcript = _filter_text(text, self.whisper_filters).strip()
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
        super().__init__(**kwargs)
        import whisper

        print(f'{INFO}Loading Whisper model: {model}')
        self.model = whisper.load_model(model)
        self.language = language

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


class FasterWhisper(AudioTranscriber):

    def __init__(self, model: str, language: str, **kwargs) -> None:
        super().__init__(**kwargs)
        from faster_whisper import WhisperModel

        print(f'{INFO}Loading Faster-Whisper model: {model}')
        self.model = WhisperModel(model, device='auto', compute_type='auto')
        self.language = language

    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        segments, info = self.model.transcribe(audio, language=self.language, initial_prompt=initial_prompt)
        text = ''
        tokens = []
        for segment in segments:
            text += segment.text
            tokens.extend(getattr(segment, 'tokens', None) or [])
        return text, tokens if tokens else None


class SimulStreaming(AudioTranscriber):

    def __init__(self, model: str, language: str, use_faster_whisper: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        from .simul_streaming.simulstreaming_whisper import SimulWhisperASR, SimulWhisperOnline

        fw_encoder = None
        if use_faster_whisper:
            print(f'{INFO}Loading Faster-Whisper as encoder for SimulStreaming: {model}')
            from faster_whisper import WhisperModel
            fw_encoder = WhisperModel(model, device='auto', compute_type='auto')

        print(f'{INFO}Loading SimulStreaming model: {model}')
        simulstreaming_params = {
            "language": language,
            "model": model,
            "cif_ckpt_path": None,
            "frame_threshold": 25,
            "audio_max_len": 20.0,
            "audio_min_len": 0.0,
            "segment_length": 0.5,
            "task": "transcribe",
            "beams": 1,
            "decoder_type": "greedy",
            "never_fire": False,
            "init_prompt": self.constant_prompt,
            "static_init_prompt": None,
            "max_context_tokens": None,
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
        super().__init__(**kwargs)
        print(f'{INFO}Using {model} API as transcription engine.')
        self.model = model
        self.language = language
        self.proxy = proxy

    def transcribe(self, audio: np.array, initial_prompt: str = None) -> tuple[str, list | None]:
        from openai import OpenAI
        import httpx

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

        ApiKeyPool.use_openai_api()
        client = OpenAI(http_client=httpx.Client(proxy=self.proxy))
        result = client.audio.transcriptions.create(**call_args).text
        return result, None

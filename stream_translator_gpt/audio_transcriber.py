import os
import queue
import tempfile
from abc import abstractmethod
from scipy.io.wavfile import write as write_audio

import numpy as np

from . import filters
from .common import TranslationTask, SAMPLE_RATE, LoopWorkerBase, sec2str, ApiKeyPool, INFO


def _filter_text(text: str, whisper_filters: str):
    filter_name_list = whisper_filters.split(',')
    for filter_name in filter_name_list:
        filter = getattr(filters, filter_name)
        if not filter:
            raise Exception('Unknown filter: %s' % filter_name)
        text = filter(text)
    return text


class AudioTranscriber(LoopWorkerBase):

    @abstractmethod
    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        pass

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_queue: queue.SimpleQueue[TranslationTask],
             whisper_filters: str, print_result: bool, output_timestamps: bool, whisper_task: str = 'transcribe',
             **transcribe_options):
        is_translation = whisper_task == 'translate'
        while True:
            task = input_queue.get()
            result = _filter_text(self.transcribe(task.audio, **transcribe_options), whisper_filters).strip()
            if not result:
                if print_result:
                    print('skip...')
                continue
            # Store result in appropriate field based on task type
            if is_translation:
                task.translation = result
            else:
                task.transcript = result
            if print_result:
                if output_timestamps:
                    timestamp_text = f'{sec2str(task.time_range[0])} --> {sec2str(task.time_range[1])}'
                    print(timestamp_text + ' ' + result)
                else:
                    print(result)
            output_queue.put(task)


class OpenaiWhisper(AudioTranscriber):

    def __init__(self, model: str, language: str, whisper_task: str = 'transcribe') -> None:
        import whisper

        print(f'{INFO}Loading Whisper model: {model}')
        self.model = whisper.load_model(model)
        self.language = language
        self.task = whisper_task

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        result = self.model.transcribe(audio, without_timestamps=True, language=self.language, task=self.task,
                                       **transcribe_options)
        return result.get('text')


class FasterWhisper(AudioTranscriber):

    def __init__(self, model: str, language: str, whisper_task: str = 'transcribe') -> None:
        from faster_whisper import WhisperModel

        print(f'{INFO}Loading Faster-Whisper model: {model}')
        self.model = WhisperModel(model, device='auto', compute_type='auto')
        self.language = language
        self.task = whisper_task

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        segments, info = self.model.transcribe(audio, language=self.language, task=self.task, **transcribe_options)
        transcript = ''
        for segment in segments:
            transcript += segment.text
        return transcript


class SimulStreaming(AudioTranscriber):

    def __init__(self, model: str, language: str, use_faster_whisper: bool, whisper_task: str = 'transcribe') -> None:
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
            "task": whisper_task,
            "beams": 1,
            "decoder_type": "greedy",
            "never_fire": False,
            "init_prompt": None,
            "static_init_prompt": None,
            "max_context_tokens": None,
            "logdir": None,
            "fw_encoder": fw_encoder,
        }
        asr = SimulWhisperASR(**simulstreaming_params)
        self.asr_online = SimulWhisperOnline(asr)

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        self.asr_online.init()
        self.asr_online.insert_audio_chunk(audio)
        result = self.asr_online.finish()
        return result.get('text', '')


class RemoteOpenaiTranscriber(AudioTranscriber):
    # https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=python

    def __init__(self, model: str, language: str, proxy: str, whisper_task: str = 'transcribe') -> None:
        task_name = 'translation' if whisper_task == 'translate' else 'transcription'
        print(f'{INFO}Using {model} API as {task_name} engine.')
        self.model = model
        self.language = language
        self.proxy = proxy
        self.task = whisper_task

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        from openai import OpenAI, DefaultHttpxClient
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as temp_audio_file:
                temp_file_path = temp_audio_file.name
                write_audio(temp_audio_file, SAMPLE_RATE, audio)
            with open(temp_file_path, 'rb') as audio_file:
                ApiKeyPool.use_openai_api()
                client = OpenAI(http_client=DefaultHttpxClient(proxy=self.proxy))
                if self.task == 'translate':
                    # Use translations API for translation task
                    result = client.audio.translations.create(model=self.model, file=audio_file).text
                else:
                    result = client.audio.transcriptions.create(model=self.model, file=audio_file,
                                                                language=self.language).text
            return result
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

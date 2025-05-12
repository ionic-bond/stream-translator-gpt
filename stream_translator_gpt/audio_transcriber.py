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
             whisper_filters: str, print_result: bool, output_timestamps: bool, **transcribe_options):
        while True:
            task = input_queue.get()
            task.transcribed_text = _filter_text(self.transcribe(task.audio, **transcribe_options),
                                                 whisper_filters).strip()
            if not task.transcribed_text:
                if print_result:
                    print('skip...')
                continue
            if print_result:
                if output_timestamps:
                    timestamp_text = '{} --> {}'.format(sec2str(task.time_range[0]), sec2str(task.time_range[1]))
                    print(timestamp_text + ' ' + task.transcribed_text)
                else:
                    print(task.transcribed_text)
            output_queue.put(task)


class OpenaiWhisper(AudioTranscriber):

    def __init__(self, model: str, language: str) -> None:
        import whisper

        print(f'{INFO}Loading whisper model: {model}')
        self.model = whisper.load_model(model)
        self.language = language

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        result = self.model.transcribe(audio, without_timestamps=True, language=self.language, **transcribe_options)
        return result.get('text')


class FasterWhisper(AudioTranscriber):

    def __init__(self, model: str, language: str) -> None:
        from faster_whisper import WhisperModel

        print(f'{INFO}Loading faster-whisper model: {model}')
        self.model = WhisperModel(model)
        self.language = language

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        segments, info = self.model.transcribe(audio, language=self.language, **transcribe_options)
        transcribed_text = ''
        for segment in segments:
            transcribed_text += segment.text
        return transcribed_text


class RemoteOpenaiWhisper(AudioTranscriber):
    # https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=python

    def __init__(self, language: str, proxy: str) -> None:
        self.proxy = proxy
        self.language = language

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        from openai import OpenAI, DefaultHttpxClient
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as temp_audio_file:
                temp_file_path = temp_audio_file.name
                write_audio(temp_audio_file, SAMPLE_RATE, audio)
            with open(temp_file_path, 'rb') as audio_file:
                ApiKeyPool.use_openai_api()
                client = OpenAI(http_client=DefaultHttpxClient(proxy=self.proxy))
                result = client.audio.transcriptions.create(model='whisper-1', file=audio_file, language=self.language).text
            return result
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)


class RemoteOpenaiTranscriber(AudioTranscriber):
    # https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=python

    def __init__(self, model: str, language: str, proxy: str) -> None:
        print(f'{INFO}Using {model} API as transcription engine.')
        self.model = model
        self.language = language
        self.proxy = proxy

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        from openai import OpenAI, DefaultHttpxClient
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as temp_audio_file:
                temp_file_path = temp_audio_file.name
                write_audio(temp_audio_file, SAMPLE_RATE, audio)
            with open(temp_file_path, 'rb') as audio_file:
                ApiKeyPool.use_openai_api()
                client = OpenAI(http_client=DefaultHttpxClient(proxy=self.proxy))
                result = client.audio.transcriptions.create(model=self.model, file=audio_file, language=self.language).text
            return result
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

import base64
import json
import os
import queue
from abc import abstractmethod
from scipy.io.wavfile import write as write_audio

import numpy as np
import websocket

from . import filters
from .common import TranslationTask, SAMPLE_RATE, LoopWorkerBase, sec2str, ApiKeyPool, start_daemon_thread

TEMP_AUDIO_FILE_NAME = '_whisper_api_temp.wav'


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

        print('Loading whisper model: {}'.format(model))
        self.model = whisper.load_model(model)
        self.language = language

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        result = self.model.transcribe(audio, without_timestamps=True, language=self.language, **transcribe_options)
        return result.get('text')


class FasterWhisper(AudioTranscriber):

    def __init__(self, model: str, language: str) -> None:
        from faster_whisper import WhisperModel

        print('Loading faster-whisper model: {}'.format(model))
        self.model = WhisperModel(model)
        self.language = language

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        segments, info = self.model.transcribe(audio, language=self.language, **transcribe_options)
        transcribed_text = ''
        for segment in segments:
            transcribed_text += segment.text
        return transcribed_text


class RemoteWhisper(AudioTranscriber):
    # https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=python

    def __init__(self, language: str, proxy: str) -> None:
        self.proxy = proxy
        self.language = language

    def __del__(self):
        if os.path.exists(TEMP_AUDIO_FILE_NAME):
            os.remove(TEMP_AUDIO_FILE_NAME)

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        from openai import OpenAI, DefaultHttpxClient
        with open(TEMP_AUDIO_FILE_NAME, 'wb') as audio_file:
            write_audio(audio_file, SAMPLE_RATE, audio)
        with open(TEMP_AUDIO_FILE_NAME, 'rb') as audio_file:
            ApiKeyPool.use_openai_api()
            client = OpenAI(http_client=DefaultHttpxClient(proxy=self.proxy))
            result = client.audio.transcriptions.create(model='whisper-1', file=audio_file, language=self.language).text
        os.remove(TEMP_AUDIO_FILE_NAME)
        return result


class OpenaiTranscriber(AudioTranscriber):
    # https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=python

    def __init__(self, model: str, language: str, proxy: str) -> None:
        self.model = model
        self.language = language
        self.proxy = proxy

    def __del__(self):
        if os.path.exists(TEMP_AUDIO_FILE_NAME):
            os.remove(TEMP_AUDIO_FILE_NAME)

    def transcribe(self, audio: np.array, **transcribe_options) -> str:
        from openai import OpenAI, DefaultHttpxClient
        with open(TEMP_AUDIO_FILE_NAME, 'wb') as audio_file:
            write_audio(audio_file, SAMPLE_RATE, audio)
        with open(TEMP_AUDIO_FILE_NAME, 'rb') as audio_file:
            ApiKeyPool.use_openai_api()
            client = OpenAI(http_client=DefaultHttpxClient(proxy=self.proxy))
            result = client.audio.transcriptions.create(model=self.model, file=audio_file, language=self.language).text
        os.remove(TEMP_AUDIO_FILE_NAME)
        return result


class RealtimeOpenaiTranscriber(LoopWorkerBase):
    # https://platform.openai.com/docs/guides/realtime-transcription

    def __init__(self, model: str, language: str, proxy: str) -> None:
        self.model = model
        self.language = language
        self.proxy = proxy
        self.transcription_queue = queue.SimpleQueue()
        self.ws = None
    
    def _on_open(self, ws):
        print("WebSocket 连接已打开.")
        config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.model,
                    "language": self.language
                },
                "turn_detection": {
                    "type": "server_vad"
                }
            }
        }
        ws.send(json.dumps(config))
        print("转录会话已配置.")
        self.running = True

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data["type"] == "conversation.item.input_audio_transcription.delta":
                transcript = data.get("delta")
                if transcript:
                    print(f"部分转录: {transcript}")
            elif data["type"] == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript")
                if transcript:
                    print(f"完整转录: {transcript}")
                    self.transcription_queue.put(transcript)
            elif data["type"] == "error":
                print(f"API 错误: {data}")
        except json.JSONDecodeError:
            print(f"JSON 解析错误: {message}")
        except Exception as e:
            print(f"处理消息错误: {e}")

    def _on_error(self, ws, error):
        print("OpenAI realtime transcription error: {}".format(error))

    def _on_close(self, ws, close_status_code, close_msg):
        print("OpenAI realtime transcription close, status code: {}, msg: {}".format(close_status_code, close_msg))
        exit(0)

    def send_audio_loop(self, input_queue: queue.SimpleQueue[np.array]):
        while True:
            try:
                audio = input_queue.get()
                if audio.dtype!= np.int16:
                    audio = (audio * 32767).astype(np.int16)
                encoded_data = base64.b64encode(audio.tobytes()).decode('utf-8')
                print(len(encoded_data))
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": encoded_data
                }
                if self.ws and self.ws.sock and self.ws.sock.connected:
                    self.ws.send(json.dumps(message))
                else:
                    print("WebSocket 连接不可用，无法发送音频.")
            except Exception as e:
                print(e)

    def receive_transcription_loop(self, output_queue: queue.SimpleQueue[TranslationTask], whisper_filters: str, print_result: bool):
        while True:
            transcription = self.transcription_queue.get()
            transcription = _filter_text(transcription, whisper_filters).strip()
            if not transcription:
                if print_result:
                    print('skip...')
                continue
            if print_result:
                print(transcription)
            output_queue.put(TranslationTask(transcribed_text=transcription))

    def loop(self, input_queue: queue.SimpleQueue[np.array], output_queue: queue.SimpleQueue[TranslationTask],
             whisper_filters: str, print_result: bool, **transcribe_options):
        ApiKeyPool.use_openai_api()
        header = ['Authorization: Bearer {}'.format(os.environ['OPENAI_API_KEY']), 'OpenAI-Beta: realtime=v1']
        ws_url = os.environ['OPENAI_BASE_URL'].replace('https', 'wss') + '/realtime?intent=transcription'
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_open=self._on_open,
                                         on_message=self._on_message,
                                         on_error=self._on_error,
                                         on_close=self._on_close,
                                         header=header)

        start_daemon_thread(self.receive_transcription_loop, output_queue=output_queue, whisper_filters=whisper_filters, print_result=print_result)
        start_daemon_thread(self.send_audio_loop, input_queue=input_queue)

        self.ws.run_forever()

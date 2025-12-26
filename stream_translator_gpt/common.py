import os
import re
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np

SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 512  # Requested by silero-vad >= v5
FRAME_DURATION = SAMPLES_PER_FRAME / SAMPLE_RATE

RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = "\033[32m"
BOLD = '\033[1m'
ENDC = '\033[0m'

INFO = f'{GREEN}[INFO]{ENDC} '
WARNING = f'{YELLOW}[WARNING]{ENDC} '
ERROR = f'{RED}[ERROR]{ENDC} '


class TranslationTask:

    def __init__(self, audio: np.array, time_range: tuple[float, float]):
        self.audio = audio
        self.transcript = None
        self.translation = None
        self.time_range = time_range
        self.start_time = None
        self.translation_failed = False


class LoopWorkerBase(ABC):

    @abstractmethod
    def loop(self):
        pass


def start_daemon_thread(func, *args, **kwargs):
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()
    return thread


def sec2str(second: float):
    dt = datetime.fromtimestamp(second, tz=timezone.utc)
    result = dt.strftime('%H:%M:%S')
    result += ',' + str(int(second * 10 % 10))
    return result


class ApiKeyPool():

    @classmethod
    def init(cls, openai_api_key, google_api_key):
        cls.openai_api_key_list = [key.strip() for key in openai_api_key.split(',')] if openai_api_key else None
        cls.openai_api_key_index = 0
        cls.use_openai_api()
        cls.google_api_key_list = [key.strip() for key in google_api_key.split(',')] if google_api_key else None
        cls.google_api_key_index = 0
        cls.use_google_api()

    @classmethod
    def use_openai_api(cls):
        if not cls.openai_api_key_list:
            return
        os.environ['OPENAI_API_KEY'] = cls.openai_api_key_list[cls.openai_api_key_index]
        cls.openai_api_key_index = (cls.openai_api_key_index + 1) % len(cls.openai_api_key_list)

    @classmethod
    def use_google_api(cls):
        if not cls.google_api_key_list:
            return
        os.environ['GOOGLE_API_KEY'] = cls.google_api_key_list[cls.google_api_key_index]
        cls.google_api_key_index = (cls.google_api_key_index + 1) % len(cls.google_api_key_list)


def is_url(address):
    parsed_url = urlparse(address)

    if parsed_url.scheme and parsed_url.scheme != 'file':
        if parsed_url.netloc or (parsed_url.scheme in ['mailto', 'tel', 'data']):
            return True

    if parsed_url.scheme == 'file':
        return False

    if parsed_url.netloc:
        return True

    if os.name == 'nt':
        if re.match(r'^[a-zA-Z]:[\\/]', address):
            return False
        if address.startswith('\\\\') or address.startswith('//'):
            return False
        if '\\' in address and '/' not in address:
            return False

    if address.startswith('/') or address.startswith('./') or address.startswith('../'):
        return False

    if '/' in address or (os.name == 'nt' and '\\' in address):
        if not parsed_url.scheme and not parsed_url.netloc:
            return False

    return False

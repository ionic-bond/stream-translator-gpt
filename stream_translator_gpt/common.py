import ipaddress
import os
import re
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np

from . import __version__

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
        self.context_transcripts = None
        self.translation = None
        self.time_range = time_range
        self.start_time = None
        self.translation_failed = False


class LoopWorkerBase(ABC):

    @abstractmethod
    def loop(self):
        pass


def start_daemon_thread(func, *args, **kwargs):

    def wrapper():
        try:
            func(*args, **kwargs)
        except Exception:
            output_queue = kwargs.get('output_queue', None)
            if output_queue is not None:
                output_queue.put(None)
            raise

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    return thread


def sec2str(second: float):
    dt = datetime.fromtimestamp(second, tz=timezone.utc)
    result = dt.strftime('%H:%M:%S')
    result += ',' + str(int(second * 10 % 10))
    return result


def is_ip_host(url: str):
    if not url:
        return False
    try:
        ipaddress.ip_address(urlparse(url).hostname)
        return True
    except (ValueError, TypeError):
        return False


class ClientPool:
    verify_ssl = True

    @classmethod
    def _should_verify_ssl(cls, base_url):
        if not cls.verify_ssl:
            return False
        if is_ip_host(base_url):
            print(f'{WARNING}Base URL "{base_url}" uses a bare IP, disabling TLS certificate verification for it.')
            return False
        return True

    @classmethod
    def _create_openai_clients(cls, api_key, base_url, proxy):
        clients = []
        if not api_key:
            return clients

        from openai import OpenAI
        import httpx

        verify = cls._should_verify_ssl(base_url)
        for key in api_key.split(','):
            key = key.strip()
            if not key:
                continue
            client_args = {
                'api_key': key,
                'default_headers': {
                    'User-Agent': f'stream-translator-gpt/{__version__}'
                },
                'http_client': httpx.Client(proxy=proxy, verify=verify),
            }
            if base_url:
                client_args['base_url'] = base_url
            clients.append(OpenAI(**client_args))
        return clients

    @classmethod
    def _create_google_clients(cls, api_key, base_url, proxy):
        clients = []
        if not api_key:
            return clients

        from google import genai

        http_options = {'client_args': {'verify': cls._should_verify_ssl(base_url)}}
        if proxy:
            http_options['client_args']['proxy'] = proxy
        if base_url:
            http_options['base_url'] = base_url
        for key in api_key.split(','):
            key = key.strip()
            if not key:
                continue
            clients.append(genai.Client(api_key=key, http_options=http_options))
        return clients

    @classmethod
    def init(cls,
             openai_api_key,
             openai_transcription_api_key=None,
             google_api_key=None,
             proxy=None,
             openai_base_url=None,
             openai_transcription_base_url=None,
             google_base_url=None,
             verify_ssl=True):
        cls.verify_ssl = verify_ssl

        cls._openai_clients = cls._create_openai_clients(openai_api_key, openai_base_url, proxy)
        cls._openai_index = 0

        cls._openai_transcription_clients = cls._create_openai_clients(
            openai_transcription_api_key,
            openai_transcription_base_url,
            proxy,
        )
        cls._openai_transcription_index = 0

        cls._google_clients = cls._create_google_clients(google_api_key, google_base_url, proxy)
        cls._google_index = 0

    @classmethod
    def get_openai_client(cls):
        if not cls._openai_clients:
            return None
        client = cls._openai_clients[cls._openai_index]
        cls._openai_index = (cls._openai_index + 1) % len(cls._openai_clients)
        return client

    @classmethod
    def get_openai_transcription_client(cls):
        if not cls._openai_transcription_clients:
            return None
        client = cls._openai_transcription_clients[cls._openai_transcription_index]
        cls._openai_transcription_index = (cls._openai_transcription_index + 1) % len(cls._openai_transcription_clients)
        return client

    @classmethod
    def get_google_client(cls):
        if not cls._google_clients:
            return None
        client = cls._google_clients[cls._google_index]
        cls._google_index = (cls._google_index + 1) % len(cls._google_clients)
        return client


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

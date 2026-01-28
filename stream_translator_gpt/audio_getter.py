import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import ffmpeg
import numpy as np
from scipy import signal

from .common import SAMPLE_RATE, SAMPLES_PER_FRAME, LoopWorkerBase, INFO, WARNING


def _transport(ytdlp_proc, ffmpeg_proc):
    while (ytdlp_proc.poll() is None) and (ffmpeg_proc.poll() is None):
        try:
            chunk = ytdlp_proc.stdout.read(1024)
            ffmpeg_proc.stdin.write(chunk)
        except (BrokenPipeError, OSError):
            pass
    ytdlp_proc.kill()
    ffmpeg_proc.kill()


def _open_stream(url: str, format: str, cookies: str, proxy: str, cwd: str):
    cmd = ['yt-dlp', url, '-f', format, '-o', '-', '-q']
    if cookies:
        cmd.extend(['--cookies', cookies])
    if proxy:
        cmd.extend(['--proxy', proxy])
    ytdlp_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=cwd)

    try:
        ffmpeg_process = (ffmpeg.input('pipe:', loglevel='panic').output('pipe:',
                                                                         format='f32le',
                                                                         acodec='pcm_f32le',
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(pipe_stdin=True,
                                                                                                   pipe_stdout=True))
    except ffmpeg.Error as e:
        raise RuntimeError(f'Failed to load audio: {e.stderr.decode()}') from e

    thread = threading.Thread(target=_transport, args=(ytdlp_process, ffmpeg_process))
    thread.start()
    return ffmpeg_process, ytdlp_process


class StreamAudioGetter(LoopWorkerBase):

    def __init__(self, url: str, format: str, cookies: str, proxy: str) -> None:
        self.url = url
        self.format = format
        self.cookies = cookies
        self.proxy = proxy
        self.temp_dir = tempfile.mkdtemp()
        self.ffmpeg_process = None
        self.ytdlp_process = None
        self.byte_size = round(SAMPLES_PER_FRAME * 4)  # Factor 4 comes from float32 (4 bytes per sample)

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _exit_handler(self, signum, frame):
        if self.ffmpeg_process:
            self.ffmpeg_process.kill()
        if self.ytdlp_process:
            self.ytdlp_process.kill()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        sys.exit(0)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        print(f'{INFO}Opening stream: {self.url}')
        self.ffmpeg_process, self.ytdlp_process = _open_stream(self.url, self.format, self.cookies, self.proxy,
                                                               self.temp_dir)
        while self.ffmpeg_process.poll() is None:
            in_bytes = self.ffmpeg_process.stdout.read(self.byte_size)
            if not in_bytes:
                break
            if len(in_bytes) != self.byte_size:
                continue
            audio = np.frombuffer(in_bytes, np.float32).flatten()
            output_queue.put(audio)

        self.ffmpeg_process.kill()
        if self.ytdlp_process:
            self.ytdlp_process.kill()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        output_queue.put(None)


class LocalFileAudioGetter(LoopWorkerBase):

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.ffmpeg_process = None
        self.byte_size = round(SAMPLES_PER_FRAME * 4)  # Factor 4 comes from float32 (4 bytes per sample)

    def _exit_handler(self, signum, frame):
        if self.ffmpeg_process:
            self.ffmpeg_process.kill()
        sys.exit(0)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        print(f'{INFO}Opening local file: {self.file_path}')
        try:
            self.ffmpeg_process = (ffmpeg.input(self.file_path,
                                                loglevel='panic').output('pipe:',
                                                                         format='f32le',
                                                                         acodec='pcm_f32le',
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(pipe_stdin=True,
                                                                                                   pipe_stdout=True))
        except ffmpeg.Error as e:
            raise RuntimeError(f'Failed to load audio: {e.stderr.decode()}') from e

        while self.ffmpeg_process.poll() is None:
            in_bytes = self.ffmpeg_process.stdout.read(self.byte_size)
            if not in_bytes:
                break
            if len(in_bytes) != self.byte_size:
                continue
            audio = np.frombuffer(in_bytes, np.float32).flatten()
            output_queue.put(audio)

        self.ffmpeg_process.kill()
        output_queue.put(None)


class DeviceAudioGetter(LoopWorkerBase):

    def __init__(self, device_index: int, use_mic: bool, interval: float = 0.5) -> None:
        if platform.system() == 'Windows':
            import pyaudiowpatch as pyaudio
        else:
            try:
                import pyaudio
            except ImportError as e:
                raise RuntimeError("PyAudio is not installed. Please install it to use device capture.\n"
                                   "Debian/Ubuntu/Colab: apt install portaudio19-dev && pip install pyaudio") from e

        self.pyaudio = pyaudio.PyAudio()
        self.pyaudio_module = pyaudio
        self.device_index = device_index
        self.use_mic = use_mic
        self.interval = interval
        self.stream = None

        if use_mic:
            if self.device_index is None:
                default_device = self.pyaudio.get_default_input_device_info()
                self.device_index = default_device['index']
        else:
            if platform.system() == 'Windows':
                if self.device_index is None:
                    try:
                        wasapi_info = self.pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
                        default_speakers = self.pyaudio.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                        if not default_speakers["isLoopbackDevice"]:
                            for loopback in self.pyaudio.get_loopback_device_info_generator():
                                if default_speakers["name"] in loopback["name"]:
                                    self.device_index = loopback["index"]
                                    break
                            else:
                                self.device_index = self.pyaudio.get_default_wasapi_loopback()['index']
                        else:
                            self.device_index = default_speakers["index"]
                    except (OSError, ValueError):
                        try:
                            loopback = next(self.pyaudio.get_loopback_device_info_generator())
                            self.device_index = loopback['index']
                        except StopIteration:
                            raise RuntimeError("No loopback device found.")
            else:
                if self.device_index is None:
                    for i in range(self.pyaudio.get_device_count()):
                        info = self.pyaudio.get_device_info_by_index(i)
                        if 'monitor' in info['name'].lower() and info['maxInputChannels'] > 0:
                            self.device_index = info['index']
                            break
                    else:
                        raise RuntimeError("No monitor device found for loopback capture.")

        self.device_name = self.pyaudio.get_device_info_by_index(self.device_index)['name']

    def _exit_handler(self, signum, frame):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
        sys.exit(0)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        print(f'{INFO}Recording device: {self.device_name} ({"Input" if self.use_mic else "Output"})')

        try:
            device_info = self.pyaudio.get_device_info_by_index(self.device_index)
            native_rate = int(device_info['defaultSampleRate'])
            try:
                native_channels = int(device_info['maxInputChannels'])
            except:
                native_channels = 1
            if native_channels < 1:
                native_channels = 2

            read_size = int(native_rate * self.interval)
            self.stream = self.pyaudio.open(format=self.pyaudio_module.paFloat32,
                                            channels=native_channels,
                                            rate=native_rate,
                                            input=True,
                                            input_device_index=self.device_index,
                                            frames_per_buffer=read_size)
            self.stream.start_stream()
            buffer = np.array([], dtype=np.float32)

            while self.stream.is_active():
                try:
                    in_data = self.stream.read(read_size, exception_on_overflow=False)
                    audio = np.frombuffer(in_data, dtype=np.float32)
                    if native_channels > 1:
                        audio = audio.reshape(-1, native_channels).mean(axis=1)
                    if native_rate != SAMPLE_RATE:
                        target_len = int(len(audio) * SAMPLE_RATE / native_rate)
                        audio = signal.resample(audio, target_len)
                    buffer = np.concatenate((buffer, audio))
                    while len(buffer) >= SAMPLES_PER_FRAME:
                        chunk = buffer[:SAMPLES_PER_FRAME]
                        buffer = buffer[SAMPLES_PER_FRAME:]
                        output_queue.put(chunk)
                except OSError as e:
                    print(f'{WARNING}Audio read error: {e}')
                    continue
        except Exception as e:
            print(f'{WARNING}Audio recording error: {e}')
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.pyaudio.terminate()
        output_queue.put(None)

import os
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

from .common import SAMPLE_RATE, SAMPLES_PER_FRAME, LoopWorkerBase, INFO


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

    def __init__(self, device_index: int, recording_interval: float) -> None:
        import sounddevice as sd

        if not device_index:
            device_index = sd.default.device[0]
        else:
            sd.default.device[0] = device_index
        self.device_index = device_index
        self.device_name = sd.query_devices(device_index)['name']

        self.recording_interval = recording_interval
        self.remaining_audio = np.array([], dtype=np.float32)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        print(f'{INFO}Recording device: {self.device_name}')

        import sounddevice as sd

        def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                print(status)

            audio = np.concatenate([self.remaining_audio, indata.flatten().astype(np.float32)])
            num_samples = len(audio)
            num_chunks = num_samples // SAMPLES_PER_FRAME
            remaining_samples = num_samples % SAMPLES_PER_FRAME

            for i in range(num_chunks):
                chunk = audio[i * SAMPLES_PER_FRAME:(i + 1) * SAMPLES_PER_FRAME]
                output_queue.put(chunk)

            self.remaining_audio = audio[-remaining_samples:] if remaining_samples > 0 else np.array([],
                                                                                                     dtype=np.float32)

        with sd.InputStream(samplerate=SAMPLE_RATE,
                            blocksize=round(SAMPLE_RATE * self.recording_interval),
                            device=self.device_index,
                            channels=1,
                            dtype=np.float32,
                            callback=audio_callback):
            while True:
                time.sleep(5)
        output_queue.put(None)

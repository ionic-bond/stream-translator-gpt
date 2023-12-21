import queue
import signal
import subprocess
import sys
import threading

import ffmpeg
import numpy as np

from common import SAMPLE_RATE


def _transport(ytdlp_proc, ffmpeg_proc):
    while (ytdlp_proc.poll() is None) and (ffmpeg_proc.poll() is None):
        try:
            chunk = ytdlp_proc.stdout.read(1024)
            ffmpeg_proc.stdin.write(chunk)
        except (BrokenPipeError, OSError):
            pass
    ytdlp_proc.kill()
    ffmpeg_proc.kill()


def _open_stream(url: str, direct_url: bool, format: str, cookies: str):
    if direct_url:
        try:
            process = (ffmpeg.input(
                url, loglevel="panic").output("pipe:",
                                                 format="s16le",
                                                 acodec="pcm_s16le",
                                                 ac=1,
                                                 ar=SAMPLE_RATE).run_async(pipe_stdout=True))
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return process, None

    cmd = ['yt-dlp', url, '-f', format, '-o', '-', '-q']
    if cookies:
        cmd.extend(['--cookies', cookies])
    ytdlp_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        ffmpeg_process = (ffmpeg.input("pipe:", loglevel="panic").output("pipe:",
                                                                         format="s16le",
                                                                         acodec="pcm_s16le",
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(
                                                                             pipe_stdin=True,
                                                                             pipe_stdout=True))
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    thread = threading.Thread(target=_transport, args=(ytdlp_process, ffmpeg_process))
    thread.start()
    return ffmpeg_process, ytdlp_process


class StreamAudioGetter():

    def __init__(self, url: str, direct_url: bool, format: str, cookies: str, frame_duration: float):
        print("Opening stream {}".format(url))
        self.ffmpeg_process, self.ytdlp_process = _open_stream(url, direct_url, format, cookies)
        self.byte_size = round(frame_duration * SAMPLE_RATE * 2) # Factor 2 comes from reading the int16 stream as bytes
        signal.signal(signal.SIGINT, self._exit_handler)

    def _exit_handler(self, signum, frame):
        self.ffmpeg_process.kill()
        if self.ytdlp_process:
            self.ytdlp_process.kill()
        sys.exit(0)
    
    def work(self, output_queue: queue.SimpleQueue[np.array]):
        while self.ffmpeg_process.poll() is None:
            in_bytes = self.ffmpeg_process.stdout.read(self.byte_size)
            if not in_bytes:
                break
            if len(in_bytes) != self.byte_size:
                continue
            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
            output_queue.put(audio)

        self.ffmpeg_process.kill()
        if self.ytdlp_process:
            self.ytdlp_process.kill()
        
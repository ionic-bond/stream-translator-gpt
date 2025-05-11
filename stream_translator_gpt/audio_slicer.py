import os
import queue
import torch
import warnings

import numpy as np

from .common import TranslationTask, SAMPLE_RATE, LoopWorkerBase

warnings.filterwarnings('ignore')


def _init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


class VAD:

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.model = _init_jit_model(os.path.join(current_dir, 'silero_vad.jit'))

    def is_speech(self, audio: np.array, threshold: float = 0.5, sampling_rate: int = 16000):
        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError('Audio cannot be casted to tensor. Cast it manually')
        speech_prob = self.model(audio, sampling_rate).item()
        return speech_prob >= threshold

    def reset_states(self):
        self.model.reset_states()


class AudioSlicer(LoopWorkerBase):

    def __init__(self, frame_duration: float, continuous_no_speech_threshold: float, min_audio_length: float,
                 max_audio_length: float, prefix_retention_length: float, vad_threshold: float):
        self.vad = VAD()
        self.continuous_no_speech_threshold = round(continuous_no_speech_threshold / frame_duration)
        self.min_audio_length = round(min_audio_length / frame_duration)
        self.max_audio_length = round(max_audio_length / frame_duration)
        self.prefix_retention_length = round(prefix_retention_length / frame_duration)
        self.vad_threshold = vad_threshold
        self.sampling_rate = SAMPLE_RATE
        self.audio_buffer = []
        self.prefix_audio_buffer = []
        self.speech_count = 0
        self.no_speech_count = 0
        self.continuous_no_speech_count = 0
        self.frame_duration = frame_duration
        self.counter = 0
        self.last_slice_second = 0.0

    def put(self, audio: np.array):
        self.counter += 1
        if self.vad.is_speech(audio, self.vad_threshold, self.sampling_rate):
            self.audio_buffer.append(audio)
            self.speech_count += 1
            self.continuous_no_speech_count = 0
        else:
            if self.speech_count == 0 and self.no_speech_count == 1:
                self.slice()
            self.audio_buffer.append(audio)
            self.no_speech_count += 1
            self.continuous_no_speech_count += 1
        if self.speech_count and self.no_speech_count / 4 > self.speech_count:
            self.slice()

    def should_slice(self):
        audio_len = len(self.audio_buffer)
        if audio_len < self.min_audio_length:
            return False
        if audio_len > self.max_audio_length:
            return True
        if self.continuous_no_speech_count >= self.continuous_no_speech_threshold:
            return True
        return False

    def slice(self):
        concatenate_buffer = self.prefix_audio_buffer + self.audio_buffer
        concatenate_audio = np.concatenate(concatenate_buffer)
        self.audio_buffer = []
        self.prefix_audio_buffer = concatenate_buffer[-self.prefix_retention_length:]
        self.speech_count = 0
        self.no_speech_count = 0
        self.continuous_no_speech_count = 0
        # self.vad.reset_states()
        slice_second = self.counter * self.frame_duration
        last_slice_second = self.last_slice_second
        self.last_slice_second = slice_second
        return concatenate_audio, (last_slice_second, slice_second)

    def loop(self, input_queue: queue.SimpleQueue[np.array], output_queue: queue.SimpleQueue[TranslationTask]):
        while True:
            audio = input_queue.get()
            self.put(audio)
            if self.should_slice():
                sliced_audio, time_range = self.slice()
                task = TranslationTask(audio=sliced_audio, time_range=time_range)
                output_queue.put(task)

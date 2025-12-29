import collections
import math
import os
import queue
import torch
import warnings

import numpy as np

from .common import TranslationTask, SAMPLE_RATE, FRAME_DURATION, LoopWorkerBase

warnings.filterwarnings('ignore')


def _init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


class VAD:

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        # Current silero-vad version: v6
        self.model = _init_jit_model(os.path.join(current_dir, 'silero_vad.jit'))
        self.reset_states()

    def get_speech_prob(self, audio: np.array):
        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError('Audio cannot be casted to tensor. Cast it manually')
        return self.model(audio, SAMPLE_RATE).item()

    def reset_states(self):
        self.model.reset_states()


def _get_neg_threshold(threshold: float):
    if threshold < 0.5:
        return threshold * 0.7
    else:
        return threshold - 0.15


def _get_dynamic_no_speech_threshold(audio_length: float, initial_threshold: float, target_audio_length: float):
    # Inverse Logistic Decay Function
    try:
        dynamic_threshold = initial_threshold / (1 + math.exp(1.0 * (audio_length - target_audio_length)))
    except OverflowError:
        dynamic_threshold = 0.0
    dynamic_threshold = max(0, min(initial_threshold, dynamic_threshold))
    return dynamic_threshold


class AudioSlicer(LoopWorkerBase):

    def __init__(self, min_audio_length: float, max_audio_length: float, target_audio_length: float,
                 continuous_no_speech_threshold: float, dynamic_no_speech_threshold: bool,
                 prefix_retention_length: float, vad_threshold: float, dynamic_vad_threshold: bool):
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.prefix_retention_count = round(prefix_retention_length / FRAME_DURATION)
        self.target_audio_length = target_audio_length
        self.dynamic_no_speech_threshold = dynamic_no_speech_threshold
        if self.dynamic_no_speech_threshold:
            self.initial_no_speech_threshold = continuous_no_speech_threshold * 2
        else:
            self.static_no_speech_threshold = continuous_no_speech_threshold
        self.audio_buffer = []
        self.prefix_audio_buffer = []
        self.speech_count = 0
        self.no_speech_count = 0
        self.continuous_no_speech_count = 0
        self.counter = 0
        self.last_slice_second = 0.0

        self.vad = VAD()
        self.vad_threshold = vad_threshold
        self.vad_neg_threshold = _get_neg_threshold(vad_threshold)
        self.dynamic_vad_threshold = dynamic_vad_threshold
        if self.dynamic_vad_threshold:
            self.vad_lookback_length = round(30 / FRAME_DURATION)  # 30 seconds
            self.vad_prob_buffer = collections.deque(maxlen=self.vad_lookback_length)
            self.vad_recalc_interval = round(5 / FRAME_DURATION)  # 5 seconds
            self.vad_recalc_quantile = 0.5
            self.min_vad_threshold = 0.0001
            self.max_vad_threshold = 0.6

    def put(self, audio: np.array):
        self.counter += 1
        speech_prob = self.vad.get_speech_prob(audio)
        is_speech = speech_prob > (self.vad_neg_threshold if self.speech_count else self.vad_threshold)
        if is_speech:
            self.audio_buffer.append(audio)
            self.speech_count += 1
            self.continuous_no_speech_count = 0
        else:
            if self.speech_count == 0 and self.no_speech_count == 1:
                self.slice()
            self.audio_buffer.append(audio)
            self.no_speech_count += 1
            self.continuous_no_speech_count += 1
        if self.speech_count and self.no_speech_count / 10 > self.speech_count:
            self.slice()

        if self.dynamic_vad_threshold:
            self.vad_prob_buffer.append(speech_prob)
            if self.counter >= self.vad_lookback_length and self.counter % self.vad_recalc_interval == 0:
                data = np.array(self.vad_prob_buffer)
                new_vad_threshold = np.quantile(data, self.vad_recalc_quantile, method='linear') * 0.7
                self.vad_threshold = self.vad_threshold * 0.25 + new_vad_threshold * 0.75
                self.vad_threshold = max(self.vad_threshold, self.min_vad_threshold)
                self.vad_threshold = min(self.vad_threshold, self.max_vad_threshold)
                self.vad_neg_threshold = _get_neg_threshold(self.vad_threshold)

    def should_slice(self):
        audio_length = len(self.audio_buffer) * FRAME_DURATION
        if audio_length < self.min_audio_length:
            return False
        if audio_length > self.max_audio_length:
            return True
        if self.dynamic_no_speech_threshold:
            no_speech_threshold = _get_dynamic_no_speech_threshold(audio_length, self.initial_no_speech_threshold,
                                                                   self.target_audio_length)
        else:
            no_speech_threshold = self.static_no_speech_threshold
        if self.continuous_no_speech_count * FRAME_DURATION >= no_speech_threshold:
            return True
        return False

    def slice(self):
        concatenate_buffer = self.prefix_audio_buffer + self.audio_buffer
        concatenate_audio = np.concatenate(concatenate_buffer)
        self.audio_buffer = []
        self.prefix_audio_buffer = concatenate_buffer[-self.prefix_retention_count:]
        self.speech_count = 0
        self.no_speech_count = 0
        self.continuous_no_speech_count = 0
        slice_second = self.counter * FRAME_DURATION
        last_slice_second = self.last_slice_second
        self.last_slice_second = slice_second
        return concatenate_audio, (last_slice_second, slice_second)

    def loop(self, input_queue: queue.SimpleQueue[np.array], output_queue: queue.SimpleQueue[TranslationTask]):
        vad_reset_interval = round(60 * 5 / FRAME_DURATION)  # 5 minutes
        while True:
            audio = input_queue.get()
            if audio is None:
                output_queue.put(None)
                break
            self.put(audio)
            if self.should_slice():
                sliced_audio, time_range = self.slice()
                task = TranslationTask(sliced_audio, time_range)
                output_queue.put(task)
            if self.counter % vad_reset_interval == 0:
                self.vad.reset_states()

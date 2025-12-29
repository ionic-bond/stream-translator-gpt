from .whisper_streaming.base import OnlineProcessorInterface, ASRBase

import sys
import torch

from .simul_whisper.config import AlignAttConfig
from .simul_whisper.simul_whisper import PaddedAlignAttWhisper


class SimulWhisperASR(ASRBase):

    sep = " "

    def __init__(self, language, model, cif_ckpt_path, frame_threshold, audio_max_len, audio_min_len, segment_length,
                 beams, task, decoder_type, never_fire, init_prompt, static_init_prompt, max_context_tokens, logdir,
                 fw_encoder):
        cfg = AlignAttConfig(
            model=model,
            segment_length=segment_length,
            frame_threshold=frame_threshold,
            language=language,
            audio_max_len=audio_max_len,
            audio_min_len=audio_min_len,
            cif_ckpt_path=cif_ckpt_path,
            decoder_type=decoder_type,  #"greedy" if beams==1 else "beam",
            beam_size=beams,
            task=task,
            never_fire=never_fire,
            init_prompt=init_prompt,
            max_context_tokens=max_context_tokens,
            static_init_prompt=static_init_prompt,
            logdir=logdir,
        )
        self.model = PaddedAlignAttWhisper(cfg, fw_encoder)

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("Use SimulWhisperOnline.process_iter() instead of transcribe().")

    def warmup(self, audio, init_prompt=""):
        self.model.insert_audio(audio)
        self.model.infer(True)
        self.model.refresh_segment(complete=True)

    def use_vad(self):
        print("VAD not implemented", file=sys.stderr)

    def set_translate_task(self):
        # this is not used. Translate task is set another way.
        pass


class SimulWhisperOnline(OnlineProcessorInterface):

    def __init__(self, asr):
        self.model = asr.model
        self.file = None
        self.init()

    def init(self, offset=None):
        self.audio_chunks = []
        if offset is not None:
            self.offset = offset
        else:
            self.offset = 0
        self.beg = self.offset
        self.end = self.offset

        self.audio_bufer_offset = self.offset
        self.last_ts = -1
        self.model.refresh_segment(complete=True)

        self.unicode_buffer = []  # hide incomplete unicode character for the next iteration

    def insert_audio_chunk(self, audio):
        self.audio_chunks.append(torch.from_numpy(audio))

    def timestamped_text(self, tokens, generation, prepended_len=0):
        if not generation:
            return []

        pr = generation["progress"]
        if "result" not in generation or prepended_len > 0 or self.unicode_buffer != []:
            split_words, split_tokens = self.model.tokenizer.split_to_word_tokens(tokens)
        else:
            split_words, split_tokens = generation["result"]["split_words"], generation["result"]["split_tokens"]

        frames = [p["most_attended_frames"][0] for p in pr]
        if frames and prepended_len > 0:
            a = [frames[0]] * prepended_len
            frames = a + frames

        tokens = tokens.copy()
        ret = []
        for sw, st in zip(split_words, split_tokens):
            b = None
            for stt in st:
                t, f = tokens.pop(0), frames.pop(0)
                if t != stt:
                    raise ValueError(f"Token mismatch: {t} != {stt} at frame {f}.")
                if b is None:
                    b = f
            e = f
            out = {
                'start': b * 0.02 + self.audio_bufer_offset,
                'end': e * 0.02 + self.audio_bufer_offset,
                'text': sw,
                'tokens': st
            }
            ret.append(out)
        return ret

    def hide_incomplete_unicode(self, tokens):
        """Sometimes, the last token is an imcomplete unicode character, e.g. a part of "ň" or "ř".
        Without this, the outputs can end with '�' = Unicode Replacement Character, and the next output also
        starts with '�'.
        This function hides the last incomplete unicode character and adds it in the next iteration.
        """
        if self.unicode_buffer != []:
            tokens = self.unicode_buffer + tokens
            self.unicode_buffer = []  # clear the buffer after processing
        chars, _ = self.model.tokenizer.split_tokens_on_unicode(tokens)
        if len(chars) > 0 and chars[-1].endswith('�'):
            self.unicode_buffer = tokens[-1:]  # keep the last incomplete unicode character
            return tokens[:-1]  # remove the last token, which is incomplete unicode character
        return tokens

    def process_iter(self, is_last=False):
        if len(self.audio_chunks) == 0:
            audio = None
        else:
            audio = torch.cat(self.audio_chunks, dim=0)
            if audio.shape[0] == 0:
                audio = None
            else:
                self.end += audio.shape[0] / self.SAMPLING_RATE
        self.audio_chunks = []
        self.audio_bufer_offset += self.model.insert_audio(audio)
        tokens, generation_progress = self.model.infer(is_last=is_last)

        prepended_len = len(self.unicode_buffer)
        tokens = self.hide_incomplete_unicode(tokens)

        text = self.model.tokenizer.decode(tokens)
        if len(text) == 0:
            return {}

        # word-level timestamps
        ts_words = self.timestamped_text(tokens, generation_progress, prepended_len)

        self.beg = min(word['start'] for word in ts_words)  # it should be this
        self.beg = max(self.beg,
                       self.last_ts + 0.001)  # but let's create the timestamps non-decreasing -- at least last beg + 1
        if is_last:
            e = self.end
        else:
            e = max(word['end'] for word in ts_words)
        e = max(e, self.beg + 0.001)

        self.last_ts = e

        # return (self.beg,e,text)
        return {'start': self.beg, 'end': e, 'text': text, 'tokens': tokens, 'words': ts_words}

    def finish(self):
        o = self.process_iter(is_last=True)
        self.model.refresh_segment(complete=True)
        return o

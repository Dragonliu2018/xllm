# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Audio encoding utilities for MiMo-V2.5-ASR.

Encodes audio waveforms to RVQ codes using MiMoAudioTokenizer.
Requires MiMo-V2.5-ASR repo on PYTHONPATH:
    export PYTHONPATH=/path/to/MiMo-V2.5-ASR/src:$PYTHONPATH
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency
_tokenizer_cls = None


def _get_tokenizer_cls():
    global _tokenizer_cls
    if _tokenizer_cls is None:
        from mimo_audio_tokenizer import MiMoAudioTokenizer
        _tokenizer_cls = MiMoAudioTokenizer
    return _tokenizer_cls


class MiMoAudioEncoder:
    """Encodes audio files to RVQ codes for MiMo-V2.5-ASR.

    Usage:
        encoder = MiMoAudioEncoder.from_pretrained("/path/to/MiMo-Audio-Tokenizer")
        codes = encoder.encode("audio.wav")  # [T, audio_channels]
    """

    def __init__(self, tokenizer, mel_transform, device, audio_channels=8):
        self._tokenizer = tokenizer
        self._mel_transform = mel_transform
        self._device = device
        self._audio_channels = audio_channels

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_path: str,
        device: Optional[torch.device] = None,
        audio_channels: int = 8,
    ) -> "MiMoAudioEncoder":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TokenizerCls = _get_tokenizer_cls()
        tokenizer = TokenizerCls.from_pretrained(tokenizer_path)
        tokenizer.eval().bfloat16().to(device)
        cfg = tokenizer.config
        mel_transform = MelSpectrogram(
            sample_rate=cfg.sampling_rate,
            n_fft=cfg.nfft,
            hop_length=cfg.hop_length,
            win_length=cfg.window_size,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            n_mels=cfg.n_mels,
            power=1.0,
            center=True,
        ).to(device)
        return cls(tokenizer, mel_transform, device, audio_channels)

    def encode(self, audio_path: str) -> torch.Tensor:
        """Encode audio file to RVQ codes.

        Args:
            audio_path: Path to WAV/FLAC/MP3 file.

        Returns:
            Tensor of shape [T, audio_channels] (int32, CPU).
        """
        wav, sr = torchaudio.load(audio_path)
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        target_sr = self._tokenizer.config.sampling_rate
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.to(self._device)

        n_fft = self._tokenizer.config.nfft
        chunk_samples = 30 * target_sr
        codes_list = []
        start = 0
        while start < wav.shape[-1]:
            end = min(start + chunk_samples, wav.shape[-1])
            if 0 < wav.shape[-1] - end < n_fft:
                end = wav.shape[-1]
            chunk = wav[start:end]
            if chunk.shape[-1] < n_fft:
                chunk = torch.nn.functional.pad(chunk, (0, n_fft - chunk.shape[-1]))
            mel = self._mel_transform(chunk[None, :])
            mel = torch.log(torch.clip(mel, min=1e-7)).squeeze(0).transpose(0, 1)
            with torch.no_grad():
                codes, _ = self._tokenizer.encoder.encode(
                    input_features=mel,
                    input_lens=torch.tensor([mel.size(0)], device=self._device),
                    return_codes_only=True,
                )
            codes_list.append(codes)
            start = end

        codes = torch.cat(codes_list, dim=-1).transpose(0, 1).detach().cpu()
        return codes[:, : self._audio_channels]

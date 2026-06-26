#!/usr/bin/env python3
"""MiMo-V2.5-ASR CUDA inference via xLLM.

Usage:
    export PYTHONPATH=/root/models2/MiMo-V2.5-ASR-repo/src:$PYTHONPATH
    python examples/run_mimo_asr_cuda.py \
        --model-path /root/models2/MiMo-V2.5-ASR \
        --audio-tokenizer-path /root/models2/MiMo-Audio-Tokenizer \
        --audio-path /path/to/audio.wav
"""
import argparse, os, sys, time

from xllm import LLM, SamplingParams
from xllm.pybind.audio_utils import MiMoAudioEncoder


def main():
    parser = argparse.ArgumentParser(description="MiMo-V2.5-ASR via xLLM")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio-tokenizer-path", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--audio-tag", default="")
    parser.add_argument("--devices", default="cuda:0")
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        print(f"Audio not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Encode audio -> RVQ codes
    print("Loading audio encoder...")
    t0 = time.monotonic()
    encoder = MiMoAudioEncoder.from_pretrained(args.audio_tokenizer_path)
    print(f"  Encoder loaded in {time.monotonic()-t0:.1f}s")

    print("Encoding audio...")
    t0 = time.monotonic()
    audio_codes = encoder.encode(args.audio_path)
    T = audio_codes.shape[0]
    group_size = 4
    T_groups = T // group_size
    print(f"  codes: {audio_codes.shape}, T_groups={T_groups}, time={time.monotonic()-t0:.1f}s")

    # 2. Build text prompt with <|empty|> placeholders
    empty_tok = "<|empty|>"
    audio_span = " ".join([empty_tok] * T_groups)
    prompt = (
        f"<|im_start|>user\n"
        f"{audio_span}\n"
        f"Please transcribe the speech.<|im_end|>\n"
        f"<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    )

    # 3. Load xLLM model
    print("Loading xLLM model...")
    t0 = time.monotonic()
    llm = LLM(model=args.model_path, devices=args.devices)
    print(f"  Model loaded in {time.monotonic()-t0:.1f}s")

    # 4. Generate
    print("Generating...")
    t0 = time.monotonic()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    elapsed = time.monotonic() - t0

    # 5. Decode output
    result = outputs[0].outputs[0].text
    # Clean up special tokens
    for tok in ["<|empty|>", "<|eot|>", "<|eostm|>", "<|chinese|>", "<|english|>"]:
        result = result.replace(tok, "")
    result = result.strip()

    print(f"\nResult: {result}")
    print(f"Time: {elapsed:.2f}s")
    llm.finish()


if __name__ == "__main__":
    main()

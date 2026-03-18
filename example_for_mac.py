import argparse
import random
import re
import numpy as np
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load


def clean_text(text):
    """Strip markdown formatting for TTS."""
    # Remove horizontal rules
    text = re.sub(r'\n---+\n', '\n', text)
    # Remove heading markers but keep the text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'[*_`]', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(text, max_chars=250):
    """Split text into chunks at sentence boundaries."""
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        # Split long paragraphs on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current) + len(sentence) + 1 <= max_chars:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                # If a single sentence exceeds max_chars, split on commas
                if len(sentence) > max_chars:
                    parts = re.split(r'(?<=,)\s+', sentence)
                    sub = ""
                    for part in parts:
                        if len(sub) + len(part) + 1 <= max_chars:
                            sub = (sub + " " + part).strip()
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = part
                    current = sub
                else:
                    current = sentence
        if current:
            chunks.append(current)
    return chunks


parser = argparse.ArgumentParser(description="Synthesize speech from a text file")
parser.add_argument("input_file", help="Path to text file to synthesize")
parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file (default: output.wav)")
parser.add_argument("--voice", "-v", default=None, help="Path to audio prompt WAV file for voice cloning")
parser.add_argument("--exaggeration", type=float, default=1.0)
parser.add_argument("--cfg-weight", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (0 = random)")
args = parser.parse_args()

if args.seed != 0:
    set_seed(args.seed)

with open(args.input_file, "r") as f:
    raw = f.read()

text = clean_text(raw)
chunks = chunk_text(text)
print(f"Synthesizing {len(chunks)} chunks...")

model = ChatterboxTTS.from_pretrained(device=device)

wavs = []
for i, chunk in enumerate(chunks):
    print(f"  [{i+1}/{len(chunks)}] {chunk[:70]}...")
    wav = model.generate(
        chunk,
        audio_prompt_path=args.voice if i == 0 else None,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )
    wavs.append(wav)

combined = torch.cat(wavs, dim=-1)
ta.save(args.output, combined, model.sr)
print(f"Saved to {args.output}")

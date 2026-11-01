#!/usr/bin/env python
"""Transcribe and diarize the downloaded interviews.

Usage: pipeline.py transcribe | diarize | merge

Each stage caches per-file JSON in work/, so stages can be rerun independently.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"
WORK = ROOT / "work"
OUT = ROOT / "transcripts"

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
NUM_SPEAKERS = 2
# Checked against the opening and middle of all five transcripts; labels are per-file, so recheck for new ones.
SPEAKER_NAMES = {"SPEAKER_00": "Mary", "SPEAKER_01": "Jay"}
# Whisper hallucinated these caption-style speaker tags; dropped before turns are built.
DROPPED_WORDS = re.compile(r"(DAVID|ABELSON|JANE|WALSH)\W*|-+")
# Whisper mishearings, applied in order at merge time. Raw output stays in work/.
# These fix transcription errors only, not factual slips by the speakers.
CORRECTIONS = [
    ("M.S. Cécile Bécaud", "Emma Cecilia Bacot"),
    ("Dessert and Emma Cicillid-Becot", "de Saussure and Emma Cecilia Bacot"),
    ("Emma Cicillid-Becot", "Emma Cecilia Bacot"),
    ("Emesis Hill Beko", "Emma Cecilia Bacot"),
    ("Emma Cecilia Bacow", "Emma Cecilia Bacot"),
    ("William Baco ", "William Bacot "),
    ("Northrup", "Northrop"),
    ("Ritchie", "Richey"),
    ("Dishossers", "de Saussures"),
    ("Mary Dessiser", "Mary de Saussure"),
    ("de Saussure's left", "de Saussures left"),
    ("Jay ghouls daughter", "Jay Gould's daughter"),
    ("J. Gould", "Jay Gould"),
    ("Google Connection", "Gould connection"),
    ("Lindhurst", "Lyndhurst"),
    ("Belport", "Bellport"),
    ("people you'll be laying in Maine", "Buell Lane and Main"),
    ("identical to you Lane", "identical to Buell Lane"),
    ("Beulain", "Buell Lane"),
    ("Bule Lane", "Buell Lane"),
    ("Beale Lane", "Buell Lane"),
    ("Beulah Lane", "Buell Lane"),
    ("Doom Cottage", "Dune Cottage"),
    ("Middle 6", "Middlesex"),
    ("Waramug", "Waramaug"),
    ("Lubeck", "Lubec"),
    ("this is Jan.", "this is Jay."),
    ("Albert Skank Cox", "Albert Schenck Cox"),
    ("sisters were Francis", "sisters were Frances"),
    ("and Francis and Aunt Helen", "and Frances and Aunt Helen"),
    ("helen and francis hall lived", "helen and frances all lived"),
    ("Aunt Mary", "Aunt Marion"),
    ("Aunt Marian", "Aunt Marion"),
    ("called Birdie", "called Bertie"),
    ("Myrtee", "Mertie"),
    ("Maude", "Maud"),
    ("maude", "maud"),
]
# Spelling hints for family names. Whisper uses this as context, not as text to transcribe.
INITIAL_PROMPT = (
    "A family history conversation with Mary Damask about the Bacot, DeSaussure, "
    "Cox, Northrop, and Gould families."
)


def sources():
    return sorted(AUDIO.glob("*.webm"))


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.rename(path)


def transcribe():
    import mlx_whisper

    for src in sources():
        out = WORK / f"{src.stem}.whisper.json"
        if out.exists():
            continue
        print(f"transcribing {src.name}", flush=True)
        result = mlx_whisper.transcribe(
            str(src),
            path_or_hf_repo=WHISPER_MODEL,
            language="en",
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=INITIAL_PROMPT,
        )
        write_json(out, result)


def to_wav(src):
    wav = WORK / f"{src.stem}.wav"
    if not wav.exists():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
        )
    return wav


def diarize():
    import soundfile
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=os.environ["HUGGING-FACE"])
    # Needs torch 2.8 on macOS 14: newer builds crash on MPS (missing MPSNDArray selector).
    pipeline.to(torch.device("mps"))

    for src in sources():
        out = WORK / f"{src.stem}.diar.json"
        if out.exists():
            continue
        print(f"diarizing {src.name}", flush=True)
        samples, sample_rate = soundfile.read(to_wav(src), dtype="float32")
        waveform = torch.from_numpy(samples).unsqueeze(0)
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=NUM_SPEAKERS)
        # exclusive_* assigns one speaker per instant, which is what word alignment needs.
        turns = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)
        ]
        write_json(out, turns)


def speaker_for(word, turns):
    """Return the speaker overlapping the word most, else the nearest turn's speaker."""
    start, end = word["start"], word["end"]
    overlap = {}
    for t in turns:
        amount = min(end, t["end"]) - max(start, t["start"])
        if amount > 0:
            overlap[t["speaker"]] = overlap.get(t["speaker"], 0) + amount
    if overlap:
        return max(overlap, key=overlap.get)
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))["speaker"]


def correct(text):
    for wrong, right in CORRECTIONS:
        text = text.replace(wrong, right)
    return text


def timestamp(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def merge():
    OUT.mkdir(exist_ok=True)
    for src in sources():
        whisper = json.loads((WORK / f"{src.stem}.whisper.json").read_text())
        turns = json.loads((WORK / f"{src.stem}.diar.json").read_text())
        info = json.loads((AUDIO / f"{src.stem}.info.json").read_text())

        blocks = []
        for segment in whisper["segments"]:
            for word in segment.get("words", []):
                if DROPPED_WORDS.fullmatch(word["word"].strip()):
                    continue
                speaker = speaker_for(word, turns)
                if not blocks or blocks[-1]["speaker"] != speaker:
                    blocks.append({"speaker": speaker, "start": word["start"], "end": word["end"], "text": ""})
                blocks[-1]["text"] += word["word"]
                blocks[-1]["end"] = word["end"]
        for block in blocks:
            block["text"] = correct(block["text"].strip())
            block["speaker"] = SPEAKER_NAMES[block["speaker"]]

        title, video_id = src.stem.rsplit("__", 1)
        write_json(OUT / f"{title}.json", {"title": info["title"], "video_id": video_id, "turns": blocks})
        lines = [f"# {info['title']}", "", f"Source: https://www.youtube.com/watch?v={video_id}", ""]
        for block in blocks:
            lines += [f"**{block['speaker']}** [{timestamp(block['start'])}]", block["text"], ""]
        (OUT / f"{title}.md").write_text("\n".join(lines))
        print(f"wrote {title}.md: {len(blocks)} turns", flush=True)


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    {"transcribe": transcribe, "diarize": diarize, "merge": merge}[sys.argv[1]]()

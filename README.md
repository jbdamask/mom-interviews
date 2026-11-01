<p align="center">
  <img src="images/mom.jpg" alt="Mary Damask laughing on a rooftop above a city, her hair blowing in the wind" width="400">
</p>

# Mary Damask interviews

Transcripts of five conversations in which Jay asks Mary Damask, Jay's mother, about her family and her childhood. The file names date the recordings to November 29, 2019, and the videos are on the [Damask family conversations YouTube channel](https://www.youtube.com/@damask-family-conversation4628). Together they run about 76 minutes.

| Transcript | Topic | Length |
|---|---|---|
| `bacot-desaussure` | The Bacot and de Saussure families, and a table from Charleston | 15:10 |
| `cousin-helen-and-the-goulds` | Helen Gould Shepard and the Gould connection | 12:32 |
| `john-burr-northrop-sr-and-family` | Mary's father and his siblings | 16:30 |
| `mary-damask-homes-growing-up` | Mary's homes in New York and East Hampton | 18:18 |
| `mary-virginia-cox-side-of-the-family` | Mary's mother and the Cox family | 13:40 |

## How the transcripts were made

The whole process ran on a Mac, and no audio went to a transcription service.

1. `yt-dlp` downloaded each video's original audio.
2. Whisper large-v3, through `mlx-whisper`, transcribed the audio with a timestamp for each word.
3. pyannote `speaker-diarization-community-1` worked out who was speaking when, with the speaker count fixed at two.
4. A merge step gave each word to the speaker who overlapped it most, then grouped the words into turns. It also applied a hand-written list of corrections to names and places that Whisper misheard, and removed speaker tags Whisper invented.

The transcripts are drafts. Nobody has compared the full text against the audio. Words near a change of speaker often land on the wrong person, which can split a sentence across two turns, and short replies like "Yeah" have the least reliable labels. Other names and words may still be wrong.

## Where to find things

| Path | Contents |
|---|---|
| `transcripts/*.md` | Readable transcripts with speakers and timestamps |
| `transcripts/*.json` | The same turns with exact start and end times |
| `transcripts/fact_checks.jsonl` | One fact check per line |
| `scripts/pipeline.py` | Transcription, diarization, merge, and the corrections list |
| `scripts/build_fact_checks.py` | The fact checks and the code that locates them |
| `requirements.txt` | Python 3.13 dependencies |
| `images/mom.jpg` | The photo of Mary at the top of this README |
| `audio/` | Downloaded audio and video metadata (gitignored) |
| `work/` | Raw Whisper and pyannote output, plus converted WAV files (gitignored) |

Whisper's uncorrected text stays in `work/*.whisper.json`.

## Running it

The diarization model is gated, so accept its [terms on Hugging Face](https://huggingface.co/pyannote/speaker-diarization-community-1) first. The script reads the access token from an environment variable named `HUGGING-FACE`, which the `with-key` shell function supplies here.

```sh
uv venv --python 3.13 .venv-py3.13
source .venv-py3.13/bin/activate
uv pip install -r requirements.txt

yt-dlp -f bestaudio --download-archive audio/archive.txt --write-info-json \
  -o "audio/%(title)s__%(id)s.%(ext)s" \
  "https://www.youtube.com/@damask-family-conversation4628/videos"

python scripts/pipeline.py transcribe
with-key HUGGING-FACE python scripts/pipeline.py diarize
python scripts/pipeline.py merge
python scripts/build_fact_checks.py
```

Each stage skips files it has already finished. Run `build_fact_checks.py` after every merge because a merge can change line numbers.

`torch` is pinned to 2.8 because newer builds crash on the GPU under macOS 14.4.

## The fact check

Claude read all five transcripts, picked out claims that public sources could confirm or contradict, and looked those up. `fact_checks.jsonl` records only the deviations it found, 20 in all: 12 labeled Mary and 8 labeled Jay. Each line gives the video, the time, a YouTube link to that moment, the transcript line, the speaker label, the quote, the deviation, and the sources.

`build_fact_checks.py` finds each quote in the transcript and in Whisper's word timings to get the line number and time, and it stops with an error unless the quote matches exactly once.

Nineteen of the 20 entries rest only on Wikipedia and the other on one news article, read through search and page-summary tools rather than always in full, so check primary records before treating a finding as settled. Family memories that left no public record went unchecked, and a claim missing from the file may still be wrong. Speaker labels carry the same errors as the transcripts. The transcripts keep what Mary and Jay said, so factual slips stay in the text and appear only in the fact-check file.

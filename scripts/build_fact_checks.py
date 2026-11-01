#!/usr/bin/env python
"""Build transcripts/fact_checks.jsonl, deriving each time and line from its quote.

Run after `pipeline.py merge`, since merging can change transcript line numbers.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
OUT = ROOT / "transcripts"

BACOT = "bacot-desaussure.mary-damask-conversation.20191129"
GOULDS = "cousin-helen-and-the-goulds.mary-damask-conversation.20191129"
HOMES = "mary-damask-homes-growing-up.mary-damask-conversations.20191129"
COX = "mary-virginia-cox-side-of-the-family.mary-damask-conversation.20191129"

RICHEY = "https://en.wikipedia.org/wiki/Thomas_Richey"
HUGUENOTS = "https://en.wikipedia.org/wiki/Huguenots"
SAUSSURE = "https://en.wikipedia.org/wiki/Horace_B%C3%A9n%C3%A9dict_de_Saussure"
JAY_GOULD = "https://en.wikipedia.org/wiki/Jay_Gould"
HELEN = "https://en.wikipedia.org/wiki/Helen_Miller_Gould"
LYNDHURST = "https://en.wikipedia.org/wiki/Lyndhurst_(mansion)"

# (transcript, quote exactly as it appears on one transcript line, deviation, sources)
CHECKS = [
    (BACOT, "they fled the Civil War back in 1865",
     "The account has Emma Cecilia Bacot coming north during the Civil War and marrying Thomas Richey afterward. "
     "Richey's biography dates the marriage to 1858, three years before the war began.",
     [RICHEY]),
    (BACOT, "He came from Ireland in 1850",
     "Thomas Richey's biography says he had settled in Pittsburgh by 1847, at least three years earlier.",
     [RICHEY]),
    (BACOT, "he came out here in 1894 when the first trains came out",
     "East Hampton station was built and opened in 1895, on June 1, by the Brooklyn and Montauk Railroad, "
     "so the first trains there came about a year later than stated.",
     ["https://en.wikipedia.org/wiki/East_Hampton_station"]),
    (BACOT, "kicked out of France by the Moors",
     "The Moors had no role. French Catholic rulers persecuted the Huguenots, most decisively Louis XIV, "
     "who revoked the Edict of Nantes in 1685 and declared Protestantism illegal.",
     [HUGUENOTS]),
    (BACOT, "variation of Catholic Catholicism",
     "Huguenots were French Protestants in the Reformed (Calvinist) tradition, not a variation of Catholicism.",
     [HUGUENOTS]),
    (BACOT, "back in maybe the 1600s",
     "Horace-Bénédict de Saussure lived 1740-1799 and climbed Mont Blanc in 1787, so this was the 1700s. "
     "The speakers revise the date to the 18th century later in the recording.",
     [SAUSSURE]),
    (GOULDS, "made four children two sons and two daughters",
     "Jay Gould had six children: four sons (George, Edwin, Howard, Frank) and two daughters (Helen, Anna).",
     [JAY_GOULD]),
    (GOULDS, "52nd Street",
     "The American Bible Society building sold around this time was at 1865 Broadway, near Columbus Circle and "
     "Lincoln Center, which the society had occupied since 1966. It was not on 52nd Street.",
     ["https://pres-outlook.org/2014/03/american-bible-society-sell-12-story-nyc-building/"]),
    (GOULDS, "That was 40-something",
     "Helen Gould Shepard died on December 21, 1938, so Christmases at her townhouse could not have been in the "
     "1940s. Mary gives her own birth year as 1933, which would make her 5 at most. These memories may be from "
     "the 1930s, or from gatherings hosted by someone else after Helen's death.",
     [HELEN]),
    (GOULDS, "the 40s, 1940s then",
     "Helen Gould Shepard died on December 21, 1938, so an event she personally arranged could not have been in "
     "the 1940s. The wedding-gown event may have been earlier, or held after her death.",
     [HELEN]),
    (GOULDS, "I think Castellani or something",
     "The name is Castellane: Anna Gould married Boniface \"Boni\" de Castellane in 1895. Her biography lists her "
     "titles as Duchess of Talleyrand and Duchess of Sagan, through her second husband, not princess.",
     ["https://en.wikipedia.org/wiki/Anna_Gould"]),
    (GOULDS, "started, I think, Monte Carlo",
     "The son is probably Frank Jay Gould, who developed casinos and hotels on the French Riviera, including at "
     "Juan-les-Pins. His biography does not connect him to founding Monte Carlo.",
     ["https://en.wikipedia.org/wiki/Frank_Jay_Gould"]),
    (GOULDS, "of the four children, the eldest son",
     "Jay Gould had six children, not four.",
     [JAY_GOULD]),
    (GOULDS, "that Jay Gould had built",
     "Jay Gould did not build Lyndhurst. It was designed in 1838 and completed in 1842, and he bought it in 1880. "
     "The bowling alley mentioned here was built in 1894, after his death in 1892.",
     [LYNDHURST, JAY_GOULD]),
    (GOULDS, "state-owned",
     "This was offered as a guess. Anna Gould donated Lyndhurst to the National Trust for Historic Preservation "
     "in 1961, rather than to the state.",
     [LYNDHURST]),
    (HOMES, "Probably the 65th",
     "The 7th Regiment Armory, now the Park Avenue Armory, occupies the block between 66th and 67th Streets.",
     ["https://en.wikipedia.org/wiki/Park_Avenue_Armory"]),
    (HOMES, "a place called Peter Cooper Community",
     "The complex is called Peter Cooper Village. The Metropolitan Life detail is correct.",
     ["https://en.wikipedia.org/wiki/Stuyvesant_Town%E2%80%93Peter_Cooper_Village"]),
    (COX, "iron is magnetic, it's paramagnetic",
     "Iron is ferromagnetic. Paramagnetic materials are magnetized only while a field is applied.",
     ["https://en.wikipedia.org/wiki/Ferromagnetism"]),
    (COX, "only about 20 years old, maybe even less",
     "This was recorded in 2019. The GPS project began in 1973, civilian use dates from the 1980s, and the system "
     "became fully operational in 1993, about 26 years earlier.",
     ["https://en.wikipedia.org/wiki/Global_Positioning_System"]),
    (COX, "the easternmost point in the U.S.",
     "West Quoddy Head in Lubec is the easternmost point of the contiguous United States, not of the entire U.S.",
     ["https://en.wikipedia.org/wiki/West_Quoddy_Head"]),
]


def normalize(text):
    return re.sub(r"[\W_]+", "", text).lower()


def locate_line(lines, quote, title):
    matches = [i for i, line in enumerate(lines) if quote in line]
    if len(matches) != 1:
        raise SystemExit(f"{title}: {quote!r} matched {len(matches)} lines")
    index = matches[0]
    label = next(line for line in reversed(lines[:index]) if line.startswith("**"))
    return index + 1, re.match(r"\*\*(\w+)\*\*", label)[1]


def locate_time(words, quote, title):
    stream, owners = "", []
    for word in words:
        piece = normalize(word["word"])
        stream += piece
        owners += [word["start"]] * len(piece)
    target = normalize(quote)
    if stream.count(target) != 1:
        raise SystemExit(f"{title}: {quote!r} matched {stream.count(target)} times in the audio timings")
    return owners[stream.index(target)]


def build():
    records = []
    for title, quote, deviation, sources in CHECKS:
        lines = (OUT / f"{title}.md").read_text().split("\n")
        transcript = json.loads((OUT / f"{title}.json").read_text())
        whisper = json.loads(next(WORK.glob(f"{title}__*.whisper.json")).read_text())
        words = [w for s in whisper["segments"] for w in s.get("words", [])]
        line, speaker = locate_line(lines, quote, title)
        seconds = int(locate_time(words, quote, title))
        records.append({
            "video": title,
            "time": f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}",
            "video_url": f"https://www.youtube.com/watch?v={transcript['video_id']}&t={seconds}s",
            "transcript": f"transcripts/{title}.md",
            "transcript_line": line,
            "speaker": speaker,
            "quote": quote,
            "deviation": deviation,
            "sources": sources,
        })
    (OUT / "fact_checks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"wrote {len(records)} fact checks")


if __name__ == "__main__":
    build()

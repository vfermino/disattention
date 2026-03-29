"""
Download mythic corpus and prepare dataset for the DisattentionFormer.

Primary source: Wikisource (en.wikisource.org) via MediaWiki API.
Fallback: Archive.org for texts not on Wikisource (Gilgamesh, Vico, Popol Vuh).
Last resort: Gutenberg (unreliable, slow).

Corpus order is intentional -- the mythic sequence matters.
shuffle=False is the most important and strangest choice.
The archetypal contamination of one text over the next is the central mechanism.

The sequence follows progressive archetypal activation:
creation -> hero -> shadow -> individuation -> Self.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Wikisource API helpers
# ---------------------------------------------------------------------------

WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
HEADERS = {"User-Agent": "DisattentionFormer/1.0 (mythic corpus builder)"}


def _wikisource_parse(page: str, timeout: int = 90) -> str:
    """Fetch a single Wikisource page and return cleaned plain text."""
    params = {
        "action": "parse",
        "page": page,
        "prop": "text",
        "format": "json",
        "disabletoc": "1",
    }
    r = requests.get(WIKISOURCE_API, params=params, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    html = data.get("parse", {}).get("text", {}).get("*", "")
    return _html_to_text(html)


def _html_to_text(html: str) -> str:
    """Convert Wikisource HTML to clean plain text."""
    text = html
    # Remove <style> and <script> blocks
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    # Remove CSS .mw-parser-output rules that leak through
    text = re.sub(r"\.mw-parser-output[^{]*\{[^}]*\}", "", text)
    # Convert breaks and paragraphs to newlines
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"</div>", "\n", text)
    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    entity_map = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&mdash;": "--",
        "&ndash;": "-",
        "&rsquo;": "'",
        "&lsquo;": "'",
        "&rdquo;": '"',
        "&ldquo;": '"',
        "&hellip;": "...",
        "&lt;": "<",
        "&gt;": ">",
    }
    for ent, char in entity_map.items():
        text = text.replace(ent, char)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"&#\d+;", "", text)
    # Clean whitespace
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _wikisource_subpages(prefix: str, limit: int = 500) -> list[str]:
    """List all subpages under a Wikisource prefix, sorted."""
    params = {
        "action": "query",
        "list": "allpages",
        "apprefix": prefix + "/",
        "apnamespace": "0",
        "aplimit": str(limit),
        "format": "json",
    }
    r = requests.get(WIKISOURCE_API, params=params, timeout=30, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    titles = [p["title"] for p in data.get("query", {}).get("allpages", [])]
    return titles


def download_wikisource_work(
    main_page: str,
    dest: Path,
    subpage_filter: callable = None,
    subpage_sort_key: callable = None,
) -> str:
    """
    Download a complete work from Wikisource.

    If the work is split into subpages (chapters/books), fetches and
    concatenates them in order.

    Args:
        main_page: The Wikisource page title (e.g. "Ulysses (1922)")
        dest: Local file path to cache the result
        subpage_filter: Optional filter function for subpage titles
        subpage_sort_key: Optional sort key for subpage ordering
    """
    if dest.exists():
        print(f"    Already cached: {dest.name}")
        return dest.read_text(encoding="utf-8")

    # Try to get subpages first
    subpages = _wikisource_subpages(main_page)

    if subpage_filter:
        subpages = [s for s in subpages if subpage_filter(s)]
    if subpage_sort_key:
        subpages.sort(key=subpage_sort_key)

    if subpages:
        print(f"    Fetching {len(subpages)} subpages of {main_page}...")
        parts = []
        for i, sp in enumerate(subpages):
            try:
                text = _wikisource_parse(sp)
                if len(text) > 50:  # skip empty/stub pages
                    parts.append(text)
                if (i + 1) % 10 == 0:
                    print(f"      {i + 1}/{len(subpages)} done...")
                # Be polite to the server
                time.sleep(0.3)
            except Exception as e:
                print(f"      Warning: failed to fetch {sp}: {e}")
        text = "\n\n".join(parts)
    else:
        # Single page work
        print(f"    Fetching single page: {main_page}...")
        text = _wikisource_parse(main_page)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Sort helpers for subpage ordering
# ---------------------------------------------------------------------------


def _chapter_num(title: str) -> int:
    """Extract chapter/book number from a subpage title."""
    # Match patterns like "Chapter 1", "Book XII", "Canto 31", etc.
    m = re.search(
        r"(?:Chapter|Book|Canto|Part|Section|Hymn)\s+(\d+|[IVXLCDM]+)\s*$",
        title,
        re.IGNORECASE,
    )
    if m:
        val = m.group(1)
        # Try Roman numeral conversion
        try:
            return _roman_to_int(val)
        except ValueError:
            return int(val)
    # Try trailing number
    m = re.search(r"(\d+)\s*$", title)
    if m:
        return int(m.group(1))
    return 0


def _roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to integer."""
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s = s.upper().strip()
    if not s or not all(c in roman for c in s):
        raise ValueError(f"Not a Roman numeral: {s}")
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and roman[s[i]] < roman[s[i + 1]]:
            result -= roman[s[i]]
        else:
            result += roman[s[i]]
    return result


# ---------------------------------------------------------------------------
# Archive.org / Gutenberg fallback helpers
# ---------------------------------------------------------------------------


def download_from_archive(url: str, dest: Path) -> str:
    """Download from Internet Archive with generous timeout."""
    if dest.exists():
        print(f"    Already cached: {dest.name}")
        return dest.read_text(encoding="utf-8")

    print(f"    Downloading from Archive.org: {url}")
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    text = response.text
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return text


def download_gutenberg(url: str, dest: Path, max_retries: int = 3) -> str:
    """Download from Gutenberg with retry logic (fallback only)."""
    if dest.exists():
        print(f"    Already cached: {dest.name}")
        return dest.read_text(encoding="utf-8")

    for attempt in range(max_retries):
        try:
            print(f"    Gutenberg fallback: {url} (attempt {attempt + 1})")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            text = response.text
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            return text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"    Retry in {wait}s... ({e})")
                time.sleep(wait)
            else:
                raise


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    start_patterns = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"Produced by",
    ]
    start_idx = 0
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_idx = match.end()
            next_nl = text.find("\n", start_idx)
            if next_nl != -1:
                start_idx = next_nl + 1
            break

    end_patterns = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
        r"End of (?:the )?Project Gutenberg",
    ]
    end_idx = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_idx = match.start()
            break

    return text[start_idx:end_idx].strip()


def clean_ocr_text(text: str) -> str:
    """Clean up Archive.org OCR artifacts."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) <= 2 and not stripped.isalpha():
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Corpus definition -- mythic order
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/mythic")


def _download_genesis(dest: Path) -> str:
    """Genesis from Wikisource (King James Bible)."""
    return download_wikisource_work(
        "Bible (King James)/Genesis",
        dest,
        subpage_filter=lambda t: "Chapter" in t,
        subpage_sort_key=_chapter_num,
    )


def _download_rigveda(dest: Path) -> str:
    """Rig Veda hymns from Wikisource."""
    return download_wikisource_work(
        "The Rig Veda",
        dest,
        subpage_sort_key=_chapter_num,
    )


def _download_popol_vuh(dest: Path) -> str:
    """Popol Vuh from Archive.org (Lewis Spence translation, 1908)."""
    archive_url = (
        "https://archive.org/download/popolvuh00spenuoft/popolvuh00spenuoft_djvu.txt"
    )
    try:
        raw = download_from_archive(archive_url, dest)
        return clean_ocr_text(raw)
    except Exception as e:
        print(f"    Warning: Could not download Popol Vuh: {e}")
        print(f"    Using placeholder...")
        text = _popol_vuh_placeholder()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return text


def _download_iliad(dest: Path) -> str:
    """Iliad (Butler translation) from Wikisource."""
    return download_wikisource_work(
        "The Iliad of Homer (Butler)",
        dest,
        subpage_filter=lambda t: "Book" in t,
        subpage_sort_key=_chapter_num,
    )


def _download_odyssey(dest: Path) -> str:
    """Odyssey (Butler translation) from Wikisource."""
    return download_wikisource_work(
        "The Odyssey (Butler)",
        dest,
        subpage_filter=lambda t: "Book" in t,
        subpage_sort_key=_chapter_num,
    )


def _download_gilgamesh(dest: Path) -> str:
    """Epic of Gilgamesh from Archive.org."""
    archive_url = (
        "https://archive.org/download/"
        "TheEpicofGilgamesh_201606/"
        "The%20Epic%20of%20Gilgamesh.txt"
    )
    try:
        raw = download_from_archive(archive_url, dest)
        return clean_ocr_text(raw)
    except Exception as e:
        print(f"    Warning: Could not download Gilgamesh: {e}")
        print(f"    Using placeholder...")
        text = _gilgamesh_placeholder()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return text


def _download_homeric_hymns(dest: Path) -> str:
    """Homeric Hymns from Wikisource (Hesiod, the Homeric Hymns and Homerica)."""
    return download_wikisource_work(
        "Hesiod, the Homeric Hymns and Homerica",
        dest,
        subpage_filter=lambda t: "Hymn" in t,
        subpage_sort_key=_chapter_num,
    )


def _download_dante(dest: Path) -> str:
    """Divine Comedy (Longfellow 1867) from Wikisource."""
    return download_wikisource_work(
        "Divine Comedy (Longfellow 1867)",
        dest,
        subpage_filter=lambda t: "Canto" in t,
        subpage_sort_key=lambda t: (
            0 if "Volume 1" in t else 1 if "Volume 2" in t else 2,
            _chapter_num(t),
        ),
    )


def _download_grimm(dest: Path) -> str:
    """Grimm's Household Tales (Edwardes 1912) from Wikisource."""
    text = download_wikisource_work(
        "Grimm's Household Tales (Edwardes)",
        dest,
    )
    if len(text) < 1000:
        # Fallback to Gutenberg
        print(f"    Wikisource too short, trying Gutenberg...")
        gutenberg_dest = dest.parent / "grimm_gutenberg.txt"
        raw = download_gutenberg(
            "https://www.gutenberg.org/cache/epub/2591/pg2591.txt",
            gutenberg_dest,
        )
        text = strip_gutenberg(raw)
        dest.write_text(text, encoding="utf-8")
    return text


def _download_blake(dest: Path) -> str:
    """Blake prophetic works from Wikisource."""
    parts = []

    # Marriage of Heaven and Hell
    try:
        text = download_wikisource_work(
            "The Marriage of Heaven and Hell",
            dest.parent / "blake_marriage.txt",
        )
        parts.append(text)
    except Exception as e:
        print(f"    Warning: Blake Marriage failed: {e}")

    # Songs of Innocence and of Experience (1826 edition, Copy Z)
    try:
        text = download_wikisource_work(
            "Songs of Innocence and of Experience (1826)",
            dest.parent / "blake_songs.txt",
        )
        parts.append(text)
    except Exception as e:
        print(f"    Warning: Blake Songs failed: {e}")

    if parts:
        combined = "\n\n".join(parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(combined, encoding="utf-8")
        return combined
    else:
        print(f"    Using placeholder for Blake...")
        text = _blake_placeholder()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return text


def _download_vico(dest: Path) -> str:
    """Vico's New Science from Archive.org (Bergin & Fisch 1948)."""
    archive_url = (
        "https://archive.org/download/"
        "newscienceofgiam030174mbp/"
        "newscienceofgiam030174mbp_djvu.txt"
    )
    try:
        raw = download_from_archive(archive_url, dest)
        return clean_ocr_text(raw)
    except Exception as e:
        print(f"    Warning: Could not download Vico: {e}")
        print(f"    Using placeholder...")
        text = _vico_placeholder()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return text


def _download_joyce_dubliners(dest: Path) -> str:
    """Dubliners from Wikisource."""
    return download_wikisource_work("Dubliners", dest)


def _download_joyce_portrait(dest: Path) -> str:
    """A Portrait of the Artist as a Young Man from Wikisource."""
    return download_wikisource_work(
        "A Portrait of the Artist as a Young Man",
        dest,
        subpage_filter=lambda t: "Chapter" in t,
        subpage_sort_key=_chapter_num,
    )


def _download_joyce_ulysses(dest: Path) -> str:
    """Ulysses from Wikisource."""
    return download_wikisource_work(
        "Ulysses (1922)",
        dest,
        subpage_filter=lambda t: "Chapter" in t,
        subpage_sort_key=_chapter_num,
    )


def _download_joyce_exiles(dest: Path) -> str:
    """Exiles from Wikisource."""
    return download_wikisource_work("Exiles (Joyce)", dest)


def _download_joyce_chamber_music(dest: Path) -> str:
    """Chamber Music from Wikisource."""
    return download_wikisource_work("Chamber Music", dest)


# Corpus in mythic order -- each entry is (name, downloader_function)
CORPUS_PIPELINE = [
    # Divine age: cosmogony, creation
    ("genesis", _download_genesis),
    ("rigveda_hymns", _download_rigveda),
    ("popol_vuh", _download_popol_vuh),
    # Heroic age: epic, quest
    ("iliad", _download_iliad),
    ("odyssey", _download_odyssey),
    ("gilgamesh", _download_gilgamesh),
    ("homeric_hymns", _download_homeric_hymns),
    # Human age: descent, knowledge, reckoning
    ("dante_commedia", _download_dante),
    ("grimm_tales", _download_grimm),
    ("blake_prophetic", _download_blake),
    # Vico: the theory of the cycle itself
    ("vico_new_science", _download_vico),
    # Joyce: the ricorso
    ("joyce_dubliners", _download_joyce_dubliners),
    ("joyce_portrait", _download_joyce_portrait),
    ("joyce_ulysses", _download_joyce_ulysses),
    ("joyce_exiles", _download_joyce_exiles),
    ("joyce_chamber_music", _download_joyce_chamber_music),
]


def load_cached_corpus() -> list[tuple[str, str]]:
    """
    Load corpus texts from cached files in mythic order.
    Only reads from disk -- no network requests.
    Returns list of (name, cleaned_text) tuples.
    """
    # Mythic order must be preserved
    MYTHIC_ORDER = [n for n, _ in CORPUS_PIPELINE]
    corpus = []

    for name in MYTHIC_ORDER:
        dest = DATA_DIR / f"{name}.txt"
        if not dest.exists():
            print(f"  [{name}] not found, skipping")
            continue
        text = dest.read_text(encoding="utf-8")
        if len(text) < 100:
            print(f"  [{name}] too short ({len(text)} chars), skipping")
            continue
        print(f"  [{name}] {len(text):,} characters")
        corpus.append((name, text))

    return corpus


def download_corpus() -> list[tuple[str, str]]:
    """
    Download all corpus texts in mythic order.
    Returns list of (name, cleaned_text) tuples.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus = []

    for name, downloader in tqdm(CORPUS_PIPELINE, desc="Corpus"):
        dest = DATA_DIR / f"{name}.txt"
        print(f"\n  [{name}]")

        try:
            text = downloader(dest)
        except Exception as e:
            print(f"    ERROR: Failed to download {name}: {e}")
            continue

        if len(text) < 100:
            print(f"    Warning: {name} too short ({len(text)} chars), skipping")
            continue

        print(f"    {name}: {len(text):,} characters")
        corpus.append((name, text))

    return corpus


# ---------------------------------------------------------------------------
# Placeholder texts for when downloads fail
# ---------------------------------------------------------------------------


def _gilgamesh_placeholder() -> str:
    return """
He who saw the Deep, the country's foundation,
who knew the proper ways, was wise in all matters!
Gilgamesh, who saw the Deep, the country's foundation,
who knew the proper ways, was wise in all matters!
He explored everywhere the seats of power,
he knew the totality of wisdom about all things.
He saw what was secret, discovered what was hidden,
he brought back a tale of before the Deluge.
He came a far road, was weary, found peace,
and set all his labours on a tablet of stone.

Enkidu, the wild man of the steppe, born of silence and clay,
ran with the gazelles and knew no name for himself.
Until the temple woman came and taught him bread and shame.
Then he walked toward the city, toward the walls,
toward the one who would call him brother and then lose him.

They went to the forest of cedars.
They killed Humbaba, the guardian, whose voice was the flood.
The gods were angry. The gods are always angry.
Enkidu sickened. Enkidu died.

Gilgamesh wept. He said: shall I too not die?
He walked to the edge of the world.
He found Utnapishtim, the one who survived the flood.
Utnapishtim said: the gods gave me life but they will not give it to you.
There is a plant at the bottom of the sea.
Gilgamesh dove. He found the plant.
A serpent ate it while he slept.
He went home. He looked at the walls of Uruk.
He said: these walls are enough.
""".strip()


def _vico_placeholder() -> str:
    return """
The nature of peoples is first crude, then severe, then benign,
then delicate, finally dissolute.

In the first age of the world, men were all robust giants.
They felt the sky as a great animated body, and they called it Jupiter.
Every thunderclap was his voice. Every lightning bolt his anger.
The first wisdom was poetic wisdom, which was felt and imagined
before it was reflected upon by reason.

The world of civil society has certainly been made by men,
and its principles are therefore to be found within the modifications
of our own human mind. Whoever reflects on this cannot but marvel
that the philosophers should have bent all their energies to the study
of the world of nature, which, since God made it, He alone knows;
and that they should have neglected the study of the world of nations,
or civil world, which, since men had made it, men could come to know.

There must in the nature of human institutions be a mental language
common to all nations, which uniformly grasps the substance of things
feasible in human social life and expresses it with as many diverse
modifications as these same things may have diverse aspects.

The course of the nations: the age of gods, the age of heroes,
the age of men. Then the ricorso, the return, the eternal cycle.
The nations rise from barbarism to civilization and back to barbarism again.
But the return is not repetition. It is recognition.
""".strip()


def _popol_vuh_placeholder() -> str:
    return """
This is the beginning of the Ancient Word, here in this place called Quiche.
Here we shall inscribe, we shall implant the Ancient Word,
the potential and source for everything done in the citadel of Quiche.

And here we shall take up the demonstration, revelation, and account
of how things were put in shadow and brought to light by the Maker,
Modeler, named Bearer, Begetter, Hunahpu Possum, Hunahpu Coyote,
Great White Peccary, Coati, Sovereign Plumed Serpent,
Heart of the Lake, Heart of the Sea,
plate shaper, bowl shaper.

Whatever might be is simply not there: only murmurs, ripples, in the dark,
in the night. Only the Maker, Modeler alone, Sovereign Plumed Serpent,
the Bearers, Begetters are in the water, a glittering light.

And then the earth arose because of them, it was simply their word
that brought it forth. For the forming of the earth they said Earth.
It arose suddenly, just like a cloud, like a mist, now forming, unfolding.

The wooden people were made; they talked, they made words,
but they had no hearts and no minds. They did not remember
the Heart of Sky. And so they fell. A flood was brought about
by the Heart of Sky. The faces of the wooden people were crushed.

And these are the names of the hero twins: Hunahpu and Xbalanque.
They descended to Xibalba, the place of fear, the underworld.
They played ball with the lords of death and won.
They died and were reborn. They became the sun and the moon.
""".strip()


def _blake_placeholder() -> str:
    return """
Without Contraries is no progression. Attraction and Repulsion,
Reason and Energy, Love and Hate, are necessary to Human existence.

From these contraries spring what the religious call Good and Evil.
Good is the passive that obeys Reason. Evil is the active springing from Energy.
Good is Heaven. Evil is Hell.

The voice of the Devil:
All Bibles or sacred codes have been the causes of the following Errors:
That Man has two real existing principles: Viz: a Body and a Soul.
That Energy, call'd Evil, is alone from the Body; and that Reason,
call'd Good, is alone from the Soul.
That God will torment Man in Eternity for following his Energies.

But the following Contraries to these are True:
Man has no Body distinct from his Soul; for that call'd Body is
a portion of Soul discern'd by the five Senses.
Energy is the only life, and is from the Body; and Reason is the
bound or outward circumference of Energy.
Energy is Eternal Delight.

Tyger Tyger, burning bright,
In the forests of the night;
What immortal hand or eye,
Could frame thy fearful symmetry?

I must Create a System or be enslav'd by another Man's.
I will not Reason and Compare: my business is to Create.

The Authors are in Eternity.
""".strip()


# ---------------------------------------------------------------------------
# Word-level tokenizer
# ---------------------------------------------------------------------------

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOT_TOKEN = "<eot>"  # end of text -- separates works in the corpus
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOT_TOKEN]


def tokenize_words(text: str) -> list[str]:
    """Split text into word tokens and punctuation."""
    return re.findall(r"[A-Za-z'\u2019-]+|[0-9]+|[^\s]", text)


class WordTokenizer:
    """
    Word-level tokenizer built from the corpus itself.

    Every word in the mythic corpus gets its own token.
    Bakhtin: the word is the smallest unit of speech.
    """

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def build(self, corpus: list[tuple[str, str]]) -> "WordTokenizer":
        """Build vocabulary from the full ordered corpus."""
        # Collect all words in corpus order (order of first appearance)
        seen: dict[str, None] = {}
        for _name, text in corpus:
            for word in tokenize_words(text):
                if word not in seen:
                    seen[word] = None

        # Build vocab: special tokens first, then words in order of appearance
        self.idx2word = list(SPECIAL_TOKENS) + list(seen.keys())
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.idx2word)

    @property
    def pad_id(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @property
    def eot_id(self) -> int:
        return self.word2idx[EOT_TOKEN]

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        return [self.word2idx.get(w, self.unk_id) for w in tokenize_words(text)]

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        words = [self.idx2word[i] if i < len(self.idx2word) else UNK_TOKEN for i in ids]
        # Reconstruct: no space before punctuation
        parts = []
        for w in words:
            if w in SPECIAL_TOKENS:
                continue
            if parts and len(w) == 1 and not w.isalnum():
                parts.append(w)
            else:
                if parts:
                    parts.append(" ")
                parts.append(w)
        return "".join(parts)

    def save(self, path: Path):
        """Save vocabulary to a text file (one word per line)."""
        path.write_text("\n".join(self.idx2word), encoding="utf-8")

    def load(self, path: Path) -> "WordTokenizer":
        """Load vocabulary from a text file."""
        self.idx2word = path.read_text(encoding="utf-8").split("\n")
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        return self


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------


class MythicDataset(Dataset):
    """
    Dataset with preserved mythic order.

    DO NOT shuffle between batches -- the archetypal contamination
    of one text over the next is intentional.
    """

    def __init__(self, tokens: list[int], seq_len: int = 512):
        self.seq_len = seq_len
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.n_chunks = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # input, target


def build_dataloader(
    corpus: list[tuple[str, str]],
    tokenizer: WordTokenizer,
    seq_len: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Tokenize the ordered corpus and build a DataLoader. Order is preserved."""
    tokens = []
    for name, text in corpus:
        encoded = tokenizer.encode(text)
        tokens.extend(encoded)
        tokens.append(tokenizer.eot_id)  # separator between texts
        print(f"  Tokenized {name}: {len(encoded):,} tokens")

    print(f"\nTotal corpus: {len(tokens):,} tokens")

    ds = MythicDataset(tokens, seq_len)
    print(f"Training sequences: {len(ds):,} (seq_len={seq_len})")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # NEVER shuffle -- mythic order matters
        num_workers=num_workers,
        pin_memory=False,  # MPS does not support pin_memory
        drop_last=True,
    )


if __name__ == "__main__":
    print("Loading cached corpus...")
    corpus = load_cached_corpus()
    print(f"\n{len(corpus)} texts loaded.")
    total_chars = sum(len(t) for _, t in corpus)
    print(f"Total: {total_chars:,} characters")

    print("\nBuilding word-level tokenizer...")
    tok = WordTokenizer().build(corpus)
    print(f"Vocabulary size: {tok.vocab_size:,}")

    vocab_path = DATA_DIR / "vocab.txt"
    tok.save(vocab_path)
    print(f"Vocabulary saved: {vocab_path}")

    print("\nBuilding dataloader...")
    loader = build_dataloader(corpus, tok, seq_len=512, batch_size=4)
    print("Done.")

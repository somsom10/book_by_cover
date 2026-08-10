"""
Bounding: finding where the plot description starts and ends in a blurb.

The structural fact this exploits: the junk collects at the edges. A typical
contaminated blurb is [award/bestseller preface] [plot] [about the author /
praise], so finding a start and an end boundary recovers most of the plot
without classifying every sentence.

Every rule is named, so each removal can be audited afterwards. Known limit:
bounding cannot reach marketing inside a plot sentence.
"""

import re
from collections import namedtuple

# --- parameters ---

# Below this many characters after bounding, give up and return the original.
# An emptied document is worse than a noisy one
MIN_KEPT_CHARS = 200
# same idea as a ratio: a trim removing over 60% of the blurb probably erred
MIN_KEPT_RATIO = 0.40
# a paragraph that is mostly uppercase is a heading or a shout line, not plot
_CAPS_RATIO = 0.80
_CAPS_MIN_CHARS = 20
# a short heading ending in a colon ("About the Author:")
_HEADING_MAX_CHARS = 60


# --- method 1: regexes for promotional prefaces and postscripts ---
#
# Direction matters, so the lists are separate: a preface rule means "cut
# everything up to here", a postscript rule means "cut everything from here".
# A phrase that is only boilerplate at the edge is anchored to ^ - "Praise for"
# mid-sentence can be real content.

_PREFACE_RULES = [
    # "From the #1 New York Times bestselling author of..." - introducing the
    # author, which precedes the plot and is not part of it
    ("bestselling_author",
     r"\b(#\s*1\s+)?(new york times|usa today|sunday times|wall street journal|"
     r"internationally|nationally|globally)?\s*best[\s-]?selling author\b"),
    # awards announced before the blurb, almost always as their own sentence
    ("award_winner",
     r"^\s*\W*(winner of|shortlisted for|longlisted for|finalist for|"
     r"nominated for|winner:)\b"),
    # "From the author of Gone Girl" - a pointer to another book
    ("from_the_author_of",
     r"^\s*\W*from the (#\s*1\s+)?(new york times\s+)?(best[\s-]?selling\s+)?"
     r"(author|creator|writer) of\b"),
    # a screen adaptation is a fact about the book, not about the plot
    ("now_a_film",
     r"\bnow a (major )?(motion picture|film|netflix (series|original|film)|"
     r"hbo (series|film)|major (television|tv) (series|event))\b"
     r"|\bsoon to be a (major )?(motion picture|film|series)\b"),
    # "book of the year" style tags
    ("notable_book",
     r"^\s*\W*(an? )?(new york times )?(notable book|book of the year|"
     r"best book of|editors'? choice|oprah'?s book club)\b"),
    # "The instant #1 New York Times bestseller"
    ("instant_bestseller",
     r"^\s*\W*(the )?(instant )?(#\s*1\s+)?(new york times |usa today )?"
     r"best[\s-]?seller\b"),
]

# HEADING-type postscripts: not a single junk sentence but a cut point.
# Everything after "About the Author" is biography, and everything after
# "Praise for" is review quotes - even when none of those sentences fires a rule
# on its own. This was the main failure of the first version: the backward scan
# stopped immediately because the last sentence of a biography looks innocent,
# and the heading stayed in the middle of the text unremoved
_HARD_POSTSCRIPT_RULES = [
    # the author biography - the commonest postscript
    ("about_the_author",
     r"^\s*\W*about the (author|translator|illustrator|editor|book)\b"),
    # commercial extras at the back of the volume
    ("includes_extra",
     r"\bincludes? (a |an )?(excerpt|preview|bonus|sneak peek|"
     r"reading group guide|discussion guide|readers guide)\b"),
    ("reading_guide",
     r"\b(reading (group|club) guide|discussion questions|"
     r"questions for discussion|a readers guide)\b"),
    # direct appeals to the reader to buy more - never plot
    ("dont_miss",
     r"^\s*\W*(don'?t miss|look for|also by|also available|perfect for fans of|"
     r"if you (loved|liked|enjoyed))\b"),
    ("praise_block", r"^\s*\W*praise for\b"),
]

_POSTSCRIPT_RULES = [
    # edition apparatus
    ("with_introduction",
     r"\bwith an? (new |brand[\s-]new )?(introduction|foreword|afterword|"
     r"preface|essay) by\b"),
    ("translated_by",
     r"\btranslated (from the \w+ )?by\b|^\s*\W*edited by\b"),
    ("edition_stmt",
     r"^\s*\W*(this|the) (revised |deluxe |anniversary |special |new |"
     r"expanded |updated )*(edition|printing|reissue|impression)\b"),
    # only as a short standalone sentence. "First published in 1952, this
    # classic recounts the story of Basil, a young silversmith..." is a plot
    # sentence opening with a bibliographic note; cutting it cuts the plot too
    ("first_published", r"^.{0,110}\b(first|originally) published (in|by)\b.{0,60}$"),
    ("cover_art",
     r"\bcover (art|design|illustration|photograph) by\b|\bjacket (design|art) by\b"),
    ("page_count", r"^\s*\W*\d{2,4} pages\b|\b\d{2,4} pages\.\s*$"),
]


# --- method 2a: fixed recurring patterns - publisher footers ---
#
# Legal and contact text. These fire anywhere, not only at the edge, because
# they are never plot - even when stuck in the middle.

# Found by scanning the highest register-scoring sentences that no rule caught.
# Each one is a family recurring in the corpus, not a one-off.
_JUNK_RULES = [
    # a. award and honour tags, usually a sentence with no verb at all:
    # "An ALA Best Book for Young Adults", "New York Times and Publishers
    # Weekly Bestseller"
    ("award_badge",
     r"\b(ala|hugo|nebula|pulitzer|booker|newbery|caldecott|edgar|whitbread|"
     r"costa|orange|nobel|national book|man booker|pen/faulkner)\b"
     r"[^.!?]{0,40}\b(award|prize|winner|honou?r|best books?|notable)\b"
     r"|^\W*an? [A-Z][^.!?]{0,50}\bbest books? for young adults\b"
     r"|^[^.!?]{0,60}\bbest[\s-]?seller\b\s*[.!]?\s*$"),
    # b. edition and reprint statements, far more varied than first listed:
    # "Unabridged republication of the classic 1931 edition.",
    # "Unabridged, slightly corrected reprint of the 2nd, 1957 edition."
    ("reprint_stmt",
     r"\b(unabridged|abridged|facsimile)\b[^.!?]{0,80}"
     r"\b(republication|reprint|reproduction|edition|republished)\b"
     r"|\b(republication|reprint) of the\b"
     r"|\b(reissued|republished|reprinted) (by|in|as)\b"
     r"|\bnow available in (paperback|hardcover|hardback|ebook|print)\b"
     r"|\bavailable in (paperback|hardcover|ebook)[^.!?]{0,30}for the first time\b"
     r"|\bmass market edition\b|\bdigitally (enlarged|reproduced|remastered)\b"
     r"|\bupdated (typeface|layout)\b"),
    # c. a review quote attributed in brackets or by paper name, with no dash:
    # "'superb!' (Australian SF News)", "'Entirely original' Spectator"
    ("quote_attribution",
     r"[\"'“”‘’][^\"'“”‘’]{3,}[\"'“”‘’]\s*\([^)]{3,40}\)\s*$"
     r"|[\"'“”‘’][^\"'“”‘’]{6,}[\"'“”‘’]\s+[A-Z][\w' ]{2,30}\s*$"
     r"|^\W*from \d+ stars? reviews?\b"),
    # d. pure commerce
    ("commerce",
     r"\bfree to download\b|\bplus excerpts? from\b|\bbonus (material|content)\b"
     r"|\border (your copy|now)\b|\bon sale now\b|\bbuy (it |the )?now\b"),
    # f. an award tag as a sentence fragment, no verb and no known award name:
    # "Winner, Best Books 2010", "Fehrenbach Award, Best Ethnic, Minority,
    # And Women's History Publication, 1987"
    ("award_fragment",
     r"^\W*winner\s*[,:]\s*\w"
     r"|^[^.!?]{0,60}\baward\b\s*[,:][^.!?]{0,80}\d{4}\s*[.!]?\s*$"
     r"|^\W*(one of |an? )?[A-Z][\w' ]{2,30}('s)?\s*best\b[^.!?]{0,50}"
     r"\b(books?|novels?|of \d{4})\b"),
    # g. back in print. Differs from reprint_stmt in that the publisher is the
    # grammatical subject here,
    # "The OSU Press is proud to reissue this...", "is back in print with
    # a new afterword", "This Swallow Press reissue of Ladders to Fire"
    ("back_in_print",
     r"\bback in print\b"
     r"|\bis (proud|pleased|delighted) to (reissue|present|publish|offer)\b"
     r"|\b[A-Z][\w]* (Press|Books|Publishing|House) reissue of\b"
     r"|\bnow in (paperback|hardcover|hardback)\b"
     r"|\bhas been reissued\b"),
    # h. ebook conversion text
    ("ebook_format",
     r"\bcarefully crafted ebook\b|\bformatted for your (ereader|kindle|nook)\b"
     r"|\bfunctional table of contents\b|\bactive table of contents\b"
     r"|\bconverted from its physical edition\b"),
    # i. positioning the author's career. Two conditions inside the pattern -
    # a status phrasing AND a writing context - so that "Anna is known for her
    # temper" does not match
    ("author_positioning",
     r"\bis (widely |generally |universally )?"
     r"(known|regarded|recognized|considered|celebrated|acclaimed|hailed)"
     r"\s+(for|as)\b[^.!?]{0,80}"
     r"\b(writer|author|novelist|poet|work|writing|fiction|literature|"
     r"novels?|books?|stories|storytelling|prose)\b"
     r"|\bhas established (himself|herself|themselves) as\b"
     r"|\bone of the (foremost|greatest|finest|leading|most \w+) "
     r"(writers?|authors?|novelists?|poets?|storytellers?|authorities)\b"),
    # j. the publication history of the volume
    ("publication_history",
     r"\bbecame an? (immediate |instant |international )?best[\s-]?seller\b"
     r"|\bwas adapted into (a )?(film|movie|television|tv)\b"
     r"|\bwas selected by\b[^.!?]{0,60}\bfor\b[^.!?]{0,40}\b(annual|year'?s best)\b"
     r"|\bwent on to (sell|become)\b"),
    # e. the volume's front matter listing, a table of contents not a description:
    # "Foreword to the Basic Books Paperback Edition, 1974 (Gardner);
    #  Preface (Carnap); Foreword to the Dover Edition (Gardner)."
    ("front_matter_list",
     r"^\W*(foreword|preface|introduction|contents|translator'?s note)\b"
     r"[^.!?]{0,80};"),
]


_FOOTER_RULES = [
    ("copyright", r"copyright\s*(©|\(c\))|©\s*\d{4}|\ball rights reserved\b"),
    ("printed_in", r"\bprinted (and bound )?in (the )?[A-Z]"),
    ("isbn", r"\bisbn\b"),
    ("library_of_congress", r"\blibrary of congress\b"),
    # the legal disclaimer from the copyright page
    ("fiction_disclaimer",
     r"\bis a work of fiction\b|\bany resemblance to (actual|real) (persons|events)\b"),
    ("url", r"www\.\S+|https?://\S+"),
    ("social", r"\b(visit|follow|find) (the author|us|him|her|them)?\s*"
               r".{0,40}(at |on )(twitter|instagram|facebook|www\.|http)"),
]


# --- method 3: structural and linguistic heuristics ---
#
# Connectives, punctuation and capitalisation marking a shift from plot to metadata.

# Product voice: the publisher describing the object rather than the story.
# More dangerous than the other rules - "Includes twelve stories about grief" is
# content - so it is weighted low and never fires alone in non-strict mode
_PRODUCT_VOICE_RE = re.compile(
    r"^\s*\W*(includes?|featuring|features?|contains?|presents?|offers?|"
    r"provides?|collects?|compiles?|gathers?)\b", re.I)

# A quote, a dash and an attribution - the review-quote signature. The
# attribution must be short and must end the sentence, or quotations from the
# book itself get caught. Two patterns rather than one, because re.I on a shared
# pattern would cancel the capital-letter requirement
_REVIEW_ATTR_CASED_RE = re.compile(
    r"[\"“”]\s*[-–—]{1,2}\s*[A-Z][\w.'’&\- ]{2,40}\s*$")
_REVIEW_SOURCE_RE = re.compile(
    r"[-–—]{1,2}\s*(the )?(kirkus|publishers weekly|booklist|"
    r"library journal|new york times|washington post|guardian|observer|locus|"
    r"times literary supplement|entertainment weekly|people magazine|usa today|"
    r"boston globe|san francisco chronicle|npr|the atlantic|"
    r"\w+ review of books)\b", re.I)

# Second person or imperative address. "Meet Anna" is deliberately excluded:
# that is a legitimate opening for a plot
_READER_ADDRESS_RE = re.compile(
    r"^\s*\W*(don'?t miss|get ready (for|to)|if you (loved|liked|enjoyed)|"
    r"perfect for (fans|readers|anyone)|a must[\s-]read for|"
    r"you'?ll (love|never)|prepare to be)\b", re.I)

# Edition statement. Note what is NOT here: book, novel, story, collection. The
# obvious rule - "flag any sentence whose subject is the book" - destroys
# content, since "This novel explores the life of a Nigerian immigrant in 1970s
# London" is the main plot sentence of many blurbs. The object is the signal,
# not the subject, so all three conditions are required together
_EDITION_SUBJECT_RE = re.compile(
    r"\bthis (\w+\s+)?(edition|volume|printing|reissue|impression|reprint)\b", re.I)
_EDITION_VERB_RE = re.compile(
    r"\b(includes?|contains?|features?|reprints?|restores?|corrects?|adds?|"
    r"presents?|reproduces?|incorporates?)\b", re.I)
_EDITION_OBJECT_RE = re.compile(
    r"\b(introduction|notes?|index|appendix|appendices|illustrations?|"
    r"translation|foreword|afterword|errata|revisions?|bibliography|"
    r"glossary|preface|typography|annotations?)\b", re.I)

_HEADING_RE = re.compile(r"^\s*[A-Z][^.!?]{0,%d}:\s*$" % _HEADING_MAX_CHARS)


# --- method 3b: register density, for what cannot be written as a regex ---
#
# The largest family found by reading the text has no fixed verbal pattern:
# "It's an audacious, at times hilarious story that is ultimately heartbreaking
# and unforgettable." A book noun, a pile of praise adjectives and no plot, with
# no phrase to catch - so this rule uses the register weights instead.

WEIGHTS_PATH = "keyness_word_weights.csv"
# threshold calibrated on the 60 highest-scoring sentences no rule caught
REGISTER_DENSE_MIN = 2.2
REGISTER_DENSE_MIN_WORDS = 4
# a very high score, which allows three scored words to be enough
REGISTER_DENSE_STRONG = 3.0
# a strongly negative word is evidence of plot, and vetoes the rule
PLOT_EVIDENCE_MAX = -0.40

# The thing being praised. Without a book noun, an adjective-heavy sentence is
# probably describing the plot itself - "Their monumental achievement ... led to
# each being awarded the Distinguished Flying Cross" is about people, not the
# volume
_BOOK_NOUN_RE = re.compile(
    r"\b(story|stories|novel|book|work|masterpiece|masterwork|portrait|"
    r"account|saga|study|collection|memoir|biography|read|volume|epic|"
    r"debut|prose|narrative|edition|anthology|classic)\b", re.I)

# Genre names protect a sentence even at a high register score: the surrounding
# copy is jacket text, but the genre itself is topical information, and
# "suspense rises in the 1990s" is a finding. keyness disagrees (suspense scores
# 3.96 there), so this call is a judgement rather than a measurement
_GENRE_NOUN_RE = re.compile(
    r"\b(action|suspense|adventure|mystery|romance|horror|fantasy|thriller|"
    r"western|crime|noir|gothic|satire|comedy|tragedy|science fiction|"
    r"historical|detective|espionage|dystopian)\b", re.I)

# Opens a narrative clause: "the story of HOW Wol and Weeps turn the town
# upside down" is plot wrapped in a praise sentence, and must not be removed
_NARRATIVE_CLAUSE_RE = re.compile(
    r"\b(how|when|after|before|while|who|whose|where)\b\s+\w+", re.I)

_TOKEN_RE = re.compile(r"[a-z][a-z']+")
_WEIGHTS = None


def _resolve_weights(path):
    """
    Look for the weight table beside the code as well as in the working
    directory. Resolved against cwd alone, a missing file silently disabled
    register_dense and produced a different corpus.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (path, os.path.join(here, "..", "work", path),
                 os.path.join(here, path)):
        if os.path.exists(cand):
            return cand
    return None


def load_weights(path=WEIGHTS_PATH):
    """Load the weight table once. Without it the rule is inert, and says so."""
    global _WEIGHTS
    if _WEIGHTS is None:
        found = _resolve_weights(path)
        if found is None:
            import warnings
            warnings.warn(
                f"{path} not found - the register_dense rule will not fire and "
                f"the corpus will differ from the published one. Run keyness.py "
                f"first (run_all.py does this in the right order).", stacklevel=2)
            _WEIGHTS = {}
        else:
            import csv
            with open(found, newline="", encoding="utf-8") as f:
                _WEIGHTS = {r["word"]: float(r["weight"]) for r in csv.DictReader(f)}
    return _WEIGHTS


def _lookup(word, W):
    if word in W:
        return W[word]
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""),
                     ("ed", ""), ("ly", "")):
        if word.endswith(suf):
            cand = word[:-len(suf)] + rep
            if cand in W:
                return W[cand]
    return None


def register_score(sentence):
    W = load_weights()
    if not W:
        return 0.0, 0, 0.0
    vals = [v for v in (_lookup(w, W) for w in _TOKEN_RE.findall(sentence.lower()))
            if v is not None]
    if not vals:
        return 0.0, 0, 0.0
    return sum(vals) / len(vals), len(vals), min(vals)


def is_register_dense(sentence):
    """
    Four conditions together, since each alone destroys content: a high score,
    enough words for the mean to be stable, a book noun as the object of the
    praise, and no evidence of plot.
    """
    score, n, lo = register_score(sentence)
    if score < REGISTER_DENSE_MIN:
        return False
    # three scored words suffice only at a very high score, otherwise four
    if n < 3 or (n == 3 and score < REGISTER_DENSE_STRONG):
        return False
    if lo <= PLOT_EVIDENCE_MAX:
        return False
    if not _BOOK_NOUN_RE.search(sentence):
        return False
    if _NARRATIVE_CLAUSE_RE.search(sentence) or _GENRE_NOUN_RE.search(sentence):
        return False
    return True


# --- newspaper and magazine names, as phrases rather than words ---
#
# "New York Times" is a newspaper, but york and times alone are content: "new
# york" without "times" appears in 1,384 documents as a setting. Only the phrase
# is removed, never the words - a word-level blacklist would delete New York.
_PUBLICATION_PHRASES = [
    r"#?\s*1?\s*new york times", r"n\.?y\.?\s+times", r"los angeles times",
    r"sunday times", r"times literary supplement", r"wall street journal",
    r"washington post", r"publishers?'? weekly", r"kirkus reviews?",
    r"(school )?library journal", r"\bbooklist\b", r"usa today",
    r"entertainment weekly", r"\bbook review\b", r"oprah'?s book club",
    r"\bgoodreads\b", r"amazon\.com", r"barnes ?& ?noble",
    # edition phrasings on the same subject: revised / updated.
    # One pattern rather than four, or "newly revised and updated" is half-cut
    r"(newly |fully |completely |thoroughly )?"
    r"(revis\w+|updated)( and (revis\w+|updated))?(?=\s+(edition|version|and))",
]
_PUBLICATION_RE = re.compile("|".join(_PUBLICATION_PHRASES), re.I)


def strip_publication_names(text):
    """Delete newspaper names as phrases. Returns (text, count)."""
    out, n = _PUBLICATION_RE.subn(" ", str(text))
    return re.sub(r"\s{2,}", " ", out).strip(), n


def _compile(rules):
    return [(name, re.compile(pat, re.I)) for name, pat in rules]


_PREFACE = _compile(_PREFACE_RULES)
_POSTSCRIPT = _compile(_POSTSCRIPT_RULES)
_HARD_POSTSCRIPT = _compile(_HARD_POSTSCRIPT_RULES)
_FOOTER = _compile(_FOOTER_RULES)
_JUNK = _compile(_JUNK_RULES)


# --- sentence splitting, preserving offsets ---

# Split on newlines too, not only on terminal punctuation. Auditing the second
# run showed this was the remaining source of errors: "ISBN: 9780099602019\nTHE
# SECOND WAR OF THE RACES\nHorrified by the misuse of magic..." is one sentence
# to a splitter looking for a full stop, so removing the ISBN dragged the
# opening of the plot with it
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text):
    """
    Sentences as (start, end, text), with offsets into the original string so
    removals can be logged exactly.
    """
    out, pos = [], 0
    for piece in _SENT_END_RE.split(text):
        idx = text.find(piece, pos)
        if idx < 0:
            idx = pos
        if piece.strip():
            out.append((idx, idx + len(piece), piece))
        pos = idx + len(piece)
    return out


def _is_all_caps(s):
    letters = [c for c in s if c.isalpha()]
    if len(letters) < _CAPS_MIN_CHARS:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= _CAPS_RATIO


def flag_sentence(sentence, repeated=frozenset(), strict=False):
    """
    Returns (rules that fired, kind), where kind is preface / postscript / any
    and decides from which direction the sentence may be cut.
    """
    hits, kinds = [], set()
    norm = re.sub(r"\s+", " ", sentence.strip())

    for name, rx in _PREFACE:
        if rx.search(sentence):
            hits.append(name); kinds.add("preface")
    for name, rx in _POSTSCRIPT + _HARD_POSTSCRIPT:
        if rx.search(sentence):
            hits.append(name); kinds.add("postscript")
    for name, rx in _FOOTER + _JUNK:
        if rx.search(sentence):
            hits.append(name); kinds.add("any")

    # Method 2b: a sentence repeated verbatim across different books describes
    # none of them. The set comes from the corpus rather than a hand-written
    # list, so it also finds publishers nobody listed
    if norm.lower() in repeated:
        hits.append("repeated_across_books"); kinds.add("any")

    # all_caps never fires alone. Auditing the first run showed it was eating
    # plot: "WHO KNOWS WHAT EVIL LURKS IN THE HEARTS OF MEN?" and "A LUFTWAFFE
    # ACE WHO WOULDN'T DIE..." are pulp magazine opening lines, i.e. content in
    # shouting, and only "OVER TWO AND ONE-HALF MILLION COPIES IN PRINT!" was
    # advertising. Capitals mark register, not metadata
    caps = _is_all_caps(norm)
    if _REVIEW_ATTR_CASED_RE.search(sentence) or _REVIEW_SOURCE_RE.search(sentence):
        hits.append("review_attribution"); kinds.add("any")
    if _READER_ADDRESS_RE.search(sentence):
        hits.append("reader_address"); kinds.add("any")
    if _HEADING_RE.match(norm):
        hits.append("heading_colon"); kinds.add("any")
    if (_EDITION_SUBJECT_RE.search(sentence)
            and _EDITION_VERB_RE.search(sentence)
            and _EDITION_OBJECT_RE.search(sentence)):
        hits.append("edition_statement"); kinds.add("any")

    # The two weak rules. They reinforce an existing flag rather than creating
    # one, because both mark register ("product voice", "shouting") not metadata
    if is_register_dense(sentence):
        hits.append("register_dense"); kinds.add("any")

    weak = []
    if _PRODUCT_VOICE_RE.match(sentence):
        weak.append("product_voice")
    if caps:
        weak.append("all_caps")
    if weak and (hits or strict):
        hits.extend(weak); kinds.add("any")

    if not hits:
        return [], None
    if "any" in kinds:
        return hits, "any"
    if "preface" in kinds and "postscript" in kinds:
        return hits, "any"
    return hits, kinds.pop()


# rules resting on unambiguous evidence rather than linguistic inference
_HARD_EVIDENCE = frozenset({
    "repeated_across_books", "isbn", "copyright", "url", "library_of_congress",
    "printed_in", "fiction_disclaimer", "social", "about_the_author",
    "cover_art", "page_count",
})


BoundResult = namedtuple(
    "BoundResult", "start end text removed fallback n_sentences")


def bound_summary(text, repeated=frozenset(), strict=False):
    """
    Trim the edges and return a BoundResult: boundaries, bounded text, and a log
    of every removed span with the rules that fired.

    Only the edges are trimmed and the interior is never touched, which is what
    bounding means; strict=True also drops flagged interior sentences.
    """
    text = str(text)
    sents = split_sentences(text)
    if not sents:
        return BoundResult(0, len(text), text, [], False, 0)

    flags = [flag_sentence(s, repeated, strict) for _, _, s in sents]
    n = len(sents)

    # start boundary: advance while the sentence is a preface or generic junk
    lo = 0
    while lo < n and flags[lo][1] in ("preface", "any"):
        lo += 1
    # end boundary: retreat while the sentence is a postscript or generic junk
    hi = n
    while hi > lo and flags[hi - 1][1] in ("postscript", "any"):
        hi -= 1

    # Hard cut point: the heading and everything after it go, even when the
    # following sentences look innocent. Take the earliest one that is not the
    # first sentence - a blurb that is entirely "About the Author" is better
    # left to the safety rail
    for i in range(1, hi):
        if any(rx.search(sents[i][2]) for _, rx in _HARD_POSTSCRIPT):
            hi = i
            break

    removed = []
    for i in range(n):
        inside = lo <= i < hi
        hits, kind = flags[i]
        if inside and not (strict and kind == "any" and hits):
            continue
        if not hits:
            continue
        side = "start" if i < lo else ("end" if i >= hi else "interior")
        removed.append({
            "sent_index": i, "side": side, "rules": ",".join(hits),
            "char_start": sents[i][0], "char_end": sents[i][1],
            "text": sents[i][2].strip(),
        })

    if strict:
        keep = [sents[i] for i in range(lo, hi)
                if not (flags[i][1] == "any" and flags[i][0])]
        bounded = " ".join(s for _, _, s in keep).strip()
        start = keep[0][0] if keep else sents[lo][0]
        end = keep[-1][1] if keep else sents[lo][0]
    else:
        if lo >= hi:
            bounded = ""
            start = end = sents[0][0]
        else:
            start, end = sents[lo][0], sents[hi - 1][1]
            bounded = text[start:end].strip()

    # Safety rail: a trim removing almost everything probably erred, so the
    # original is returned. One exception, learned from the log: when every
    # removal came from a hard evidence rule there is nothing to save - the 112
    # documents that hit the rail were mostly publisher text end to end, so
    # returning empty and letting the length filter drop them is better
    hard = all(set(r["rules"].split(",")) <= _HARD_EVIDENCE for r in removed)
    fallback = (len(bounded) < MIN_KEPT_CHARS
                or (len(text) and len(bounded) / len(text) < MIN_KEPT_RATIO))
    if fallback and removed and hard:
        return BoundResult(start if lo < hi else 0, end if lo < hi else 0,
                           bounded, removed, False, n)
    if fallback:
        return BoundResult(0, len(text), text, removed, True, n)
    return BoundResult(start, end, bounded, removed, False, n)


ALL_RULE_NAMES = (
    [n for n, _ in _PREFACE_RULES]
    + [n for n, _ in _POSTSCRIPT_RULES]
    + [n for n, _ in _HARD_POSTSCRIPT_RULES]
    + [n for n, _ in _FOOTER_RULES]
    + [n for n, _ in _JUNK_RULES]
    + ["register_dense"]
    + ["repeated_across_books", "all_caps", "review_attribution",
       "reader_address", "heading_colon", "edition_statement", "product_voice"]
)

"""Author identity: when two spellings are one person.

Shared by comention.py (canonicalizing mentions) and eval_genres.py (matching
graph names to genre profiles) so both stages agree on who is one author."""
import re
import unicodedata
from collections import defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")
VOWELS = set("aeiouy")

# Fragments the automatic rules can't repair, verified against the data by
# hand. "William Butler" is what the longest-match mention matcher returns for
# blurb text "William Butler Yeats" (the full form isn't in the dictionary).
CURATED_ALIASES = {
    "william butler": "W.B. Yeats",
    "yeats": "W.B. Yeats",
    # both forms are spelled out, so the initials-merge rule (which needs a
    # unique spelled form) correctly refuses to merge them on its own
    "george r. r. martin": "George R.R. Martin",
}


def author_key(s):
    """Accent-folded, punctuation/case-insensitive token identity."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return tuple(TOKEN_RE.findall(s.replace(".", "").lower()))


def given_initials(tokens):
    """Initials string for the given-name part ('jrr','tolkien') -> 'jrr'.
    A short vowel-less token is treated as run-together initials."""
    out = []
    for t in tokens[:-1]:
        if len(t) <= 4 and not set(t) & VOWELS:
            out.extend(t)          # 'jrr' -> j,r,r ; 'wb' -> w,b
        else:
            out.append(t[0])
    return "".join(out)


def is_spelled(tokens):
    """At least one given-name token written out in full (not initials)."""
    return any(len(t) >= 3 and set(t) & VOWELS for t in tokens[:-1])


def signature(k):
    """(surname, given initials) -- what an initials form and its spelled-out
    form have in common. None for a single-token key."""
    return (k[-1], given_initials(k)) if len(k) >= 2 else None


def build_canonical_map(counts):
    """Merge initials-only forms into their spelled-out form when the
    (surname, initials) signature identifies exactly one spelled form.
    Conservative: any ambiguity -> no merge. Returns {key: canonical_key}."""
    by_sig = defaultdict(list)
    for k in counts:
        if len(k) >= 2:
            by_sig[signature(k)].append(k)
    merge = {}
    for forms in by_sig.values():
        if len(forms) < 2:
            continue
        spelled = [k for k in forms if is_spelled(k)]
        if len(spelled) != 1:
            continue                      # zero or ambiguous -> leave alone
        base = spelled[0]
        for k in forms:
            if k != base:
                merge[k] = base
    return merge

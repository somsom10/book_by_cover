"""Shared filtering for the author dictionary.

Organisations, publishers and newspapers sit in the CSV `Authors` column and
are mentioned in blurbs as something other than authors. Token lists are
deliberately conservative, so real surnames (Emily Post, Terry Nation) survive."""
import re

_WS = re.compile(r"\s+")
_ALPHA = re.compile(r"[a-z]+")

STRONG_ORG_TOKENS = {
    "press", "university", "publishing", "publications", "publisher",
    "publishers", "magazine", "comics", "comic", "association", "institute",
    "council", "commission", "society", "societies", "foundation", "committee",
    "ministry", "bureau", "incorporated", "inc", "studios", "studio",
    "entertainment", "productions", "production", "media", "llc", "ltd",
    "nations", "geographic", "vatican", "department", "agency", "review",
    "books", "book", "club", "bbc", "cnn", "npr", "editorial", "news",
    "audubon", "times", "company", "group", "gazette", "herald", "tribune",
    "journal", "chronicle", "worldwide", "kennel", "programs", "program",
    "project", "projects", "ministries", "syndicate", "globe", "corps",
    "government", "illustrated", "library", "libraries", "museum", "gallery",
    "academy", "academies", "college", "congress", "administration",
    "corporation", "radio", "records", "anonymous", "international",
    "scientist", "economist", "housekeeping", "pediatrics", "prospectus",
    "guides", "workshop", "team", "fund", "scholastic", "encyclopedia",
    "dictionary", "almanac",
    # Deliberately absent because they are real authors' surnames: post
    # (Emily Post), church (Forrest Church), center (Katherine Center),
    # atlas (Nava Atlas), nation (Terry Nation), board (Robert De Board),
    # homes (A.M. Homes). Those orgs go in CURATED_STOP instead.
}

CURATED_STOP = {
    "the new yorker", "the beatles", "lonely planet", "the state",
    "founding fathers", "the founding fathers", "the un", "un", "u.n.",
    "the u.n.", "not a book", "anonymous", "various", "unknown",
    "random house", "vanity fair",
    # orgs/brands whose tokens are too person-like for STRONG_ORG_TOKENS
    "the catholic church", "church of england", "episcopal church",
    "the poetry center", "the college board", "better homes", "real simple",
    "taste of home", "american girl", "american heritage", "american poetry",
    "the american poetry", "wizards of the coast", "boy scouts of america",
    "monks of new skete", "symphony space", "barron's", "the bark",
    "the harvard lampoon", "the onion", "the wire",
    # music groups (author fields of band books)
    "the doors", "the clash", "the rolling stones", "the who",
}

# Non-literary public figures (politicians, monarchs, military, scientists,
# musicians, actors, artists, athletes, business, activists) plus stray
# fictional characters. They author the odd memoir/speech collection so they
# sit in the dictionary, but they are not literary authors. Curated from the
# most-mentioned names; extend as needed. Borderline literary figures
# (Thoreau, Emerson, Frederick Douglass, Robert Frost, Lord Byron) are kept.
NON_LITERARY_FIGURES = {
    # presidents / statesmen / monarchs / military / activists
    "abraham lincoln", "george washington", "barack obama", "thomas jefferson",
    "winston churchill", "adolf hitler", "queen victoria", "john f. kennedy",
    "ronald reagan", "george w. bush", "benjamin franklin", "bill clinton",
    "theodore roosevelt", "robert e. lee", "elizabeth i", "nelson mandela",
    "fidel castro", "martin luther king jr.", "martin luther king",
    "martin luther", "ulysses s. grant", "eleanor roosevelt", "malcolm x",
    "john adams", "andrew jackson", "margaret thatcher", "jimmy carter",
    "marco polo", "mother teresa",
    # scientists / academics / philosophers / business
    "albert einstein", "charles darwin", "sigmund freud", "isaac newton",
    "richard dawkins", "michel foucault", "william james", "karl marx",
    "bill gates",
    # musicians / actors / filmmakers / artists / athletes / TV
    "bob dylan", "john lennon", "michael jackson", "david bowie",
    "marilyn monroe", "alfred hitchcock", "woody allen", "steven spielberg",
    "orson welles", "elizabeth taylor", "andy warhol", "babe ruth",
    "muhammad ali", "jackie robinson", "oprah winfrey", "martha stewart",
    "james beard",
    # fictional characters that leaked in as "authors"
    "jack frost",
}


def clean_ws(name):
    return _WS.sub(" ", name).strip()


def is_org_author(name):
    nl = clean_ws(name).lower()
    if nl in CURATED_STOP:
        return True
    return bool(set(_ALPHA.findall(nl)) & STRONG_ORG_TOKENS)


def exclude_author(name):
    """True if `name` should be dropped from the author dictionary entirely:
    organisations/publishers (is_org_author) or non-literary public figures."""
    if is_org_author(name):
        return True
    return clean_ws(name).lower() in NON_LITERARY_FIGURES

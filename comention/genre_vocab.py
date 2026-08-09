"""Curated shelf -> category vocabulary for Goodreads `popular_shelves`.

Shelves are free text and mostly not about the book: of the 600 commonest, the
top ten are `to-read`, `currently-reading`, `favorites`, `owned`, `kindle` ...
So the vocabulary is curated, not learned. Left out are status/format/ownership
shelves, and shelves too generic to separate anything (`fiction`,
`contemporary`) -- those sit on half the catalogue and only add a constant to
every profile.

Included alongside genre proper: nationality, era and language. Not genre, but
how a large part of the catalogue is shelved, and several communities are
organised by them. The cost is that for those communities the measure partly
confirms what binds them rather than testing it independently.

A shelf naming two categories (`sci-fi-fantasy`) splits its count between them.

Exported: GENRES, GENRE_NAMES (fixed column order), SHELF_GENRE."""

GENRES = {
    "fantasy": [
        "fantasy", "high-fantasy", "epic-fantasy", "dark-fantasy",
        "sword-and-sorcery", "fantasy-fiction", "fantasia", "dragons", "magic",
        "sci-fi-fantasy", "scifi-fantasy", "sff", "sf-fantasy",
        "fantasy-sci-fi", "fantasy-scifi", "science-fiction-fantasy",
        "sci-fi-and-fantasy", "fantasy-paranormal", "paranormal-fantasy",
        "urban-fantasy", "ya-fantasy",
    ],
    "science fiction": [
        "science-fiction", "sci-fi", "scifi", "sf", "space-opera", "space",
        "aliens", "cyberpunk", "steampunk", "time-travel", "dystopia",
        "dystopian", "post-apocalyptic", "apocalyptic", "apocalypse",
        "speculative-fiction", "futuristic", "sci-fi-fantasy",
        "scifi-fantasy", "sff", "sf-fantasy", "fantasy-sci-fi",
        "fantasy-scifi", "science-fiction-fantasy", "sci-fi-and-fantasy",
        "alternate-history",
    ],
    "paranormal": [
        "paranormal", "supernatural", "urban-fantasy", "shifters", "shifter",
        "shapeshifters", "fae", "witches", "angels", "vampires", "vamps",
        "vampire", "werewolves", "fantasy-paranormal", "paranormal-fantasy",
        "paranormal-romance", "romance-paranormal", "pnr", "ya-paranormal",
    ],
    "horror": [
        "horror", "gothic", "zombies", "ghosts", "demons", "horror-thriller",
    ],
    "mystery": [
        "mystery", "mysteries", "detective", "detective-fiction", "detectives",
        "murder-mystery", "murder", "cozy-mystery", "cozy-mysteries", "cozy",
        "mystery-series", "mystery-detective", "mystery-cozy",
        "historical-mystery", "mystery-thriller", "mystery-thrillers",
        "thriller-mystery", "mystery-suspense", "mystery-crime",
        "crime-mystery",
    ],
    "crime": [
        "crime", "crime-fiction", "police-procedural", "noir", "krimi",
        "mystery-crime", "crime-mystery", "crime-thriller",
    ],
    "thriller": [
        "thriller", "thrillers", "suspense", "psychological-thriller",
        "espionage", "spy", "suspense-thriller", "thriller-suspense",
        "mystery-thriller", "mystery-thrillers", "thriller-mystery",
        "mystery-suspense", "crime-thriller", "romantic-suspense",
        "horror-thriller",
    ],
    "true crime": ["true-crime"],
    "romance": [
        "romance", "contemporary-romance", "romance-contemporary",
        "historical-romance", "romance-historical", "regency-romance",
        "regency", "harlequin", "paranormal-romance", "romance-paranormal",
        "pnr", "adult-romance", "new-adult", "friends-to-lovers",
        "alpha-male", "alpha-males", "bad-boys", "book-boyfriends",
        "virgin-heroine", "love-triangle", "menage", "m-m", "mm",
        "m-m-romance", "mm-romance", "gay-romance", "romantic-suspense",
        "ya-romance",
    ],
    "chick lit": [
        "chick-lit", "chicklit", "chic-lit", "women-s-fiction",
        "womens-fiction",
    ],
    "erotica": [
        "erotica", "erotic", "erotic-romance", "bdsm", "steamy", "smut",
        "menage",
    ],
    "historical fiction": [
        "historical-fiction", "historical", "fiction-historical",
        "historical-mystery", "historical-romance", "romance-historical",
        "alternate-history",
    ],
    "literary fiction": [
        "literary-fiction", "literary", "magical-realism", "modern-fiction",
    ],
    "classics": [
        "classics", "classic", "classic-literature", "classic-fiction",
        "classic-lit", "modern-classics", "classici", "clàssics",
    ],
    "poetry": ["poetry", "poems", "poem", "verse"],
    "plays": ["plays", "theatre", "theater", "shakespeare", "playwrights"],
    "short stories": [
        "short-stories", "short-story", "short-fiction",
        "short-story-collections", "anthology", "anthologies", "novella",
        "novellas",
    ],
    "comics": [
        "comics", "comic", "comic-books", "graphic-novel", "graphic-novels",
        "graphic", "graphic-novels-comics", "comics-graphic-novels",
        "comics-and-graphic-novels", "graphic-novels-and-comics",
        "graphic-novels-comics", "manga", "mangas", "manga-comics",
        "comics-manga", "manga-manhwa", "superheroes", "superhero",
    ],
    "children's": [
        "childrens", "children", "children-s", "childrens-books",
        "children-s-books", "children-books", "picture-books", "picture-book",
        "picturebooks", "kids", "kids-books", "kid-lit", "middle-grade",
        "chapter-books", "juvenile", "juvenile-fiction",
        "children-s-literature", "childrens-literature", "childrens-lit",
        "children-s-lit", "children-s-fiction", "storytime", "read-aloud",
        "read-alouds", "preschool", "youth",
    ],
    "young adult": [
        "young-adult", "ya", "ya-fiction", "ya-books", "ya-lit", "teen",
        "teen-fiction", "young-adult-fiction", "youngadult", "ya-fantasy",
        "ya-romance", "ya-contemporary", "ya-paranormal",
    ],
    "biography & memoir": [
        "biography", "biographies", "memoir", "memoirs", "autobiography",
        "biography-memoir", "memoir-biography", "biographies-memoirs",
        "biography-autobiography", "bio", "bio-memoir", "biographical",
    ],
    "history": [
        "history", "american-history", "european-history", "ancient-history",
        "medieval", "civil-war", "wwii", "ww2", "world-war-ii",
        "world-war-2", "holocaust", "military-history",
    ],
    "war & military": [
        "war", "military", "military-fiction", "wwii", "ww2", "world-war-ii",
        "world-war-2", "holocaust", "military-history",
    ],
    "religion": [
        "religion", "religious", "christian", "christianity",
        "christian-fiction", "theology", "faith", "bible", "inspirational",
        "spirituality", "spiritual", "buddhism", "islam", "catholic",
        "devotional", "prayer",
    ],
    "philosophy": ["philosophy", "existentialism", "ethics", "metaphysics"],
    "psychology": [
        "psychology", "mental-health", "mental-illness", "psychiatry",
    ],
    "self-help": [
        "self-help", "self-improvement", "personal-development",
        "self-development", "motivational", "productivity", "self-care",
    ],
    "business & economics": [
        "business", "economics", "finance", "management", "leadership",
        "marketing", "entrepreneurship", "investing",
    ],
    "science": [
        "science", "physics", "biology", "mathematics", "math", "astronomy",
        "evolution", "chemistry", "neuroscience", "popular-science",
        "technology", "computer-science", "programming",
    ],
    "politics": [
        "politics", "political", "political-science", "government",
        "current-events", "current-affairs",
    ],
    "society": [
        "sociology", "anthropology", "social-science", "social-issues",
        "feminism", "feminist", "gender", "cultural-studies", "journalism",
    ],
    "travel": ["travel", "travel-writing", "travelogue"],
    "food": [
        "cookbooks", "cookbook", "cooking", "food", "recipes", "culinary",
        "baking",
    ],
    "art": ["art", "photography", "architecture", "design", "art-history"],
    "music": ["music", "rock", "jazz", "classical-music", "music-history"],
    "film": ["film", "films", "movies", "cinema", "film-criticism",
             "screenplays"],
    "nature": [
        "nature", "environment", "environmental", "animals", "ecology",
        "gardening",
    ],
    "sports": [
        "sports", "baseball", "football", "basketball", "running", "cycling",
    ],
    "health": [
        "health", "fitness", "nutrition", "diet", "medicine", "medical",
    ],
    "parenting": ["parenting", "pregnancy", "motherhood"],
    "western": ["western", "westerns", "cowboys"],
    "humor": ["humor", "humour", "funny", "comedy", "satire", "humorous"],
    "essays": ["essays", "essay"],
    "mythology": [
        "mythology", "folklore", "fairy-tales", "fairy-tale", "fairytales",
        "legends", "myths", "retellings",
    ],
    "lgbtq": [
        "lgbt", "lgbtq", "glbt", "lgbtqia", "queer", "gay", "lesbian",
        "transgender", "m-m", "mm", "m-m-romance", "mm-romance",
        "gay-romance",
    ],
    "reference": [
        "reference", "education", "teaching", "textbook", "writing",
        "language", "linguistics", "dictionary", "how-to",
    ],
    "adventure": [
        "adventure", "action", "action-adventure", "survival", "pirates",
    ],

    # ---- nationality, era and language ------------------------------------
    # Only previously-unmapped shelves are claimed here, so `regency` stays
    # with romance and `medieval` with history.
    "american": [
        "american", "american-literature", "american-lit",
        "american-classics", "americana",
    ],
    "african american": ["african-american", "black-authors", "black-lit"],
    "british": [
        "british", "british-literature", "british-lit", "english",
        "english-literature", "english-lit", "scottish", "welsh",
    ],
    "irish": ["irish", "irish-literature", "irish-lit"],
    "french": ["french", "french-literature", "french-lit"],
    "russian": ["russian", "russian-literature", "russian-lit"],
    "german": ["german", "german-literature", "german-lit", "germany"],
    "italian": ["italian", "italian-literature", "italian-lit"],
    "spanish": ["spanish", "spanish-literature"],
    "latin american": [
        "latin-america", "latin-american", "latin-american-literature",
    ],
    "japanese": ["japanese", "japanese-literature", "japanese-lit"],
    "asian": [
        "asian", "asian-literature", "chinese", "korean", "indian", "india",
    ],
    "scandinavian": [
        "scandinavian", "scandinavia", "nordic", "swedish", "norwegian",
        "danish", "finnish",
    ],
    "canadian": ["canadian", "canada", "canadian-literature"],
    "australian": ["australian", "australia", "new-zealand"],
    "african": ["african", "africa", "african-literature"],
    "jewish": ["jewish", "israeli", "judaica"],
    "middle eastern": [
        "arabic", "arab", "turkish", "iranian", "persian", "middle-east",
    ],
    "greek": ["greek"],
    "dutch": ["dutch"],
    "european": ["europe", "european", "european-literature"],
    "18th century": ["18th-century", "1700s"],
    "19th century": ["19th-century", "1800s", "victorian"],
    "20th century": [
        "20th-century", "1900s", "modernism", "1910s", "1920s", "1930s",
        "1940s", "1950s", "1960s", "1970s", "1980s", "1990s",
    ],
    "21st century": ["21st-century", "2000s", "2010s"],
    "ancient": ["ancient", "antiquity", "classical-antiquity"],
    "translated": [
        "translated", "translation", "translations", "in-translation",
        "world-literature", "world-lit",
    ],
}

GENRE_NAMES = sorted(GENRES)

SHELF_GENRE = {}
for _g, _shelves in GENRES.items():
    for _s in _shelves:
        if _g not in SHELF_GENRE.setdefault(_s, []):
            SHELF_GENRE[_s].append(_g)

# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: MIT

import argparse
import logging
import pickle
import random
import string
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from random import Random
from typing import Callable, Dict, Optional, Sequence, cast

from wikibaseintegrator import wbi_core, wbi_datatype, wbi_functions

from common import create_login_instance


def random_string(n):
    return "".join(random.choices(string.ascii_letters, k=n))


def format_list(items):
    """
    >>> format_list(['A', 'B'])
    'A and B'
    >>> format_list(['A', 'B', 'C'])
    'A, B and C'
    """
    assert len(items) >= 2
    return ", ".join(items[:-1]) + " and " + items[-1]


class Language(Enum):
    ENGLISH = "Q1860"
    SWEDISH = "Q9027"


class LexicalCategory(Enum):
    ADJ = "Q34698"
    NOUN = "Q1084"
    VERB = "Q24905"
    ADVERB = "Q380057"


class LexemeType(Enum):
    VERBAL_NOUN = "Q1350145"
    ABSOLUTE_ADJ = "Q332375"
    AGENT_NOUN = "Q1787727"


class Lexeme:
    def __init__(self, qid, lemma=None, category: Optional[LexicalCategory] = None):
        self.qid = qid.removeprefix("http://www.wikidata.org/entity/")
        self.lemma = lemma
        self.category = category

    def __str__(self):
        return f"Lexeme({self.qid}, {self.lemma}, {self.category})"


HistoryDict = Dict[str, tuple[datetime, bool]]


class History:
    """
    Stores information about previously processed lexemes:
    - Successfully matched lexemes are skipped forever. This way errors can be
      corrected by humans and the bot won't try to reinsert incorrect
      information.
    - Unmatched lexemes will be processed again after some time to see if
      matching works this time. This reduces repeated queries.
    """

    def __init__(self, filename: str):
        history_dir = Path("history")
        history_dir.mkdir(exist_ok=True)
        self.path = history_dir / filename
        try:
            with self.path.open("rb") as f:
                self.items: HistoryDict = pickle.load(f)
        except FileNotFoundError:
            self.items = {}
        # Store changes separately and only in the end commit these to the
        # history. This way multiple tasks may try to match the same lexeme.
        self.changes: HistoryDict = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        items = {**self.items, **self.changes}
        with self.path.open("wb") as f:
            pickle.dump(items, f)

    def add(self, lexeme: Lexeme, matched: bool, now: datetime = None):
        if now is None:
            now = datetime.now(timezone.utc)
        # Commit matched lexeme right away to the history, so that we don't try
        # to match them more than once.
        if matched:
            self.items[lexeme.qid] = (now, matched)
        else:
            self.changes[lexeme.qid] = (now, matched)

    def __contains__(self, lexeme) -> bool:
        if lexeme.qid not in self.items:
            return False
        last_checked, matched = self.items[lexeme.qid]
        if matched:
            return True
        now = datetime.now(timezone.utc)
        r = Random(f"{lexeme.qid} {last_checked}")
        return (now - last_checked).days < r.randint(14, 28)


class Result:
    def __init__(
        self,
        lexeme: Lexeme,
        parts: Sequence[Optional[Lexeme]],
        types: Sequence[LexemeType] = None,
    ):
        self.lexeme = lexeme
        self.parts = parts
        self.types = types if types is not None else []


class Task:
    def __init__(
        self,
        language: Language,
        categories: Sequence[LexicalCategory],
        transform: Callable[[str], Result],
        include=None,
        exclude=None,
        name=None,
    ):
        self.language = language
        self.categories = categories
        self.transform = transform
        self.include = include
        self.exclude = exclude
        self.name = name

    def _search_lexemes(self, limit: int):
        query = "SELECT ?lexeme ?lemma ?category WHERE {\n"
        query += f"  ?lexeme dct:language wd:{self.language.value};\n"
        query += f"    wikibase:lexicalCategory ?category;\n"
        query += "    wikibase:lemma ?lemma.\n"
        query += (
            "  FILTER(?category IN ("
            + ",".join(f"wd:{category.value}" for category in self.categories)
            + "))\n"
        )

        if self.include:
            query += f'  FILTER(REGEX(?lemma, "{self.include}"))\n'
        if self.exclude:
            query += f'  FILTER(!REGEX(?lemma, "{self.exclude}"))\n'

        # Ignore lexemes with existing combines (P5238) claims. These might be
        # previously added by this bot or humans.
        query += "  MINUS { ?lexeme wdt:P5238 []. }\n"

        # Randomize rows using custom randomization instead of RAND function.
        # This will make it sure that the order is really random and embedding
        # a random string to the query will bypass any caching. For more
        # information, see https://byabbe.se/2020/09/17/getting-random-results-in-sparql
        random = random_string(10)
        query += f'  BIND(SHA512(CONCAT("{random}", STR(?lexeme))) AS ?random)\n'
        query += "}\n"
        query += f"ORDER BY ?random\n"

        # Query extra lexemes to fill the limit because some lexemes may be
        # skipped later if no matching lexeme is found.
        query += f"LIMIT {10*limit}"

        data = wbi_functions.execute_sparql_query(query)
        lexemes = []
        for row in data["results"]["bindings"]:
            lexemes.append(
                Lexeme(
                    row["lexeme"]["value"],
                    row["lemma"]["value"],
                    LexicalCategory(
                        row["category"]["value"].removeprefix(
                            "http://www.wikidata.org/entity/"
                        )
                    ),
                )
            )
        return lexemes

    def execute(self, limit: int, history: History):
        i = 0
        for lexeme in self._search_lexemes(limit):
            if i == limit:
                break
            if lexeme in history:
                continue
            result = self.transform(lexeme)
            if result.parts and all(result.parts):
                yield result
                i += 1
                history.add(lexeme, matched=True)
            else:
                history.add(lexeme, matched=False)


tasks = []


def task(**kwargs):
    def inner(fn):
        tasks.append(Task(name=fn.__name__, **kwargs, transform=fn))

    return inner


def find_lexeme(
    lemma: str, language: Language, categories: list[LexicalCategory]
) -> Optional[Lexeme]:
    """Search a single lexeme with the specified lemma."""

    cats = ", ".join(f"wd:{cat.value}" for cat in categories)
    query = "SELECT ?lexeme ?category WHERE {\n"
    query += f"  ?lexeme dct:language wd:{language.value};\n"
    query += f"    wikibase:lexicalCategory ?category;\n"
    query += "    wikibase:lemma ?lemma.\n"
    query += f'  FILTER(?category IN ({cats}) && STR(?lemma) = "{lemma}")\n'
    query += "}\n"
    query += "LIMIT 2"

    data = wbi_functions.execute_sparql_query(query)
    results = data["results"]["bindings"]

    # To play it safe, let's continue only if we found a single lexeme.
    if len(results) != 1:
        return None

    qid = results[0]["lexeme"]["value"]
    category = LexicalCategory(
        results[0]["category"]["value"].removeprefix("http://www.wikidata.org/entity/")
    )
    return Lexeme(qid, lemma, category)


# "unbounded" → "un-" + "bounded"
# "underived" → "un-" + "derived"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.ADJ],
    include="^un...",
)
def en_un_adj(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L15649", "un-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("un"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "unbox" → "un-" + "box"
# "underive" → "un-" + "derive"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.VERB],
    include="^un...",
)
def en_un_verb(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L15649", "un-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("un"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "defuse" → "de-" + "fuse"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.VERB],
    include="^de......",
)
def en_de_verb(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L35199", "de-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("de"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "disconnect" → "dis-" + "connect"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.VERB],
    include="^dis......",
)
def en_dis_verb(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L29593", "dis-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("dis"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "misunderstand" → "mis-" + "understand"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.VERB],
    include="^mis......",
)
def en_mis_verb(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L613650", "mis-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("mis"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "antioxidant" → "anti-" + "oxidant"
# "anticlimactic" → "anti-" + "climactic"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
    include="^anti-?......",
)
def en_anti(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L29591", "anti-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("anti").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "counterculture" → "counter-" + "culture"
# "counterclockwise" → "counter-" + "clockwise"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
    include="^counter-?......",
)
def en_counter(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L36419", "counter-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("counter").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "contradistinction" → "contra-" + "distinction"
# "contralateral" → "contra-" + "lateral"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
    include="^contra-?......",
)
def en_contra(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L36418", "contra-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("contra").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "protohistory" → "proto-" + "history"
# "protoacademic" → "proto-" + "academic"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
    include="^proto-?......",
)
def en_proto(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L615092", "proto-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("proto").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "overlook" → "over-" + "look"
# "overkind" → "over-" + "kind"
# "overlord" → "over-" + "lord"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.VERB, LexicalCategory.ADJ, LexicalCategory.NOUN],
    include="^over-?...",
)
def en_over(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L618499", "over-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("over").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "restless" → "rest" + "-less"
@task(language=Language.ENGLISH, categories=[LexicalCategory.ADJ], include="...less$")
def en_less(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("less"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L303186", "-less"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "awkwardness" → "awkward" + "-ness"
# "happiness" → "happy" + "-ness"
@task(language=Language.ENGLISH, categories=[LexicalCategory.NOUN], include="...ness$")
def en_ness(lexeme: Lexeme) -> Result:
    lemma = lexeme.lemma.removesuffix("ness")
    if lemma[-1] == "i":
        lemma = lemma[:-1] + "y"
    parts = [
        find_lexeme(
            lemma=lemma,
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ],
        ),
        Lexeme("L269834", "-ness"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "guitarist" → "guitar" + "-ist"
# "surrealist" → "surreal" + "-ist"
@task(language=Language.ENGLISH, categories=[LexicalCategory.NOUN], include="...ist$")
def en_ist(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ist"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        ),
        Lexeme("L29847", "-ist"),
    ]
    return Result(lexeme=lexeme, parts=parts, types=[LexemeType.AGENT_NOUN])


# "alcoholism" → "alcohol" + "-ism"
# "surrealism" → "surreal" + "-ism"
@task(language=Language.ENGLISH, categories=[LexicalCategory.NOUN], include="...ism$")
def en_ism(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ism"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        ),
        Lexeme("L29596", "-ism"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "peaceful" → "peace" + "-ful"
# "beautiful" → "beauty" + "-ful"
@task(language=Language.ENGLISH, categories=[LexicalCategory.ADJ], include="...ful$")
def en_ful_adj(lexeme: Lexeme) -> Result:
    lemma = lexeme.lemma.removesuffix("ful")
    if lemma[-1] == "i":
        lemma = lemma[:-1] + "y"
    parts = [
        find_lexeme(
            lemma=lemma,
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L7893", "-ful"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "handful" → "hand" + "-ful"
@task(language=Language.ENGLISH, categories=[LexicalCategory.NOUN], include="...ful$")
def en_ful_noun(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ful"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L592127", "-ful"),
    ]
    return Result(lexeme=lexeme, parts=parts)


@task(language=Language.ENGLISH, categories=[LexicalCategory.ADJ], include="...able$")
def en_able(lexeme: Lexeme) -> Result:
    parts = [
        # "educable" → "educate" + "-able"
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("able") + "ate",
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        )
        # "movable" → "move" + "-able"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("able") + "e",
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        )
        # "breakable" → "break" + "-able"
        # "fashionable" → "fashion" + "-able"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("able"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB, LexicalCategory.NOUN],
        ),
        Lexeme("L457381", "-able"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "manly" → "man" + "-ly"
# "daily" → "day" + "-ly"
@task(language=Language.ENGLISH, categories=[LexicalCategory.ADJ], include="...ly$")
def en_ly_adj(lexeme: Lexeme) -> Result:
    lemma = lexeme.lemma.removesuffix("ly")
    if lemma[-1] == "i":
        lemma = lemma[:-1] + "y"
    parts = [
        find_lexeme(
            lemma=lemma,
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L592203", "-ly"),
    ]
    return Result(lexeme=lexeme, parts=parts)


@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.ADVERB],
    include="...ly$",
)
def en_ly_adverb(lexeme: Lexeme) -> Result:
    if lexeme.lemma.endswith("ally"):
        # "mythically" → "mythical" + "-ly"
        stem = find_lexeme(
            lemma=lexeme.lemma.removesuffix("ly"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ],
        )
        if stem:
            return Result(lexeme=lexeme, parts=[stem, Lexeme("L28890", "-ly")])

        # "basically" → "basic" + "-ally"
        stem = find_lexeme(
            lemma=lexeme.lemma.removesuffix("ally"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ],
        )
        return Result(lexeme=lexeme, parts=[stem, Lexeme("L592202", "-ally")])

    # "suddenly" → "sudden" + "-ly"
    lemma = lexeme.lemma.removesuffix("ly")

    # "easily" → "easy" + "-ly"
    if lemma[-1] == "i":
        lemma = lemma[:-1] + "y"
    # "fully" → "full" + "-ly"
    elif lemma[-1] == "l":
        lemma += "l"

    parts = [
        find_lexeme(
            lemma=lemma,
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ],
        ),
        Lexeme("L28890", "-ly"),
    ]
    return Result(lexeme=lexeme, parts=parts)


@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN],
    include="...ion$",
)
def en_ion(lexeme: Lexeme) -> Result:
    lemma = lexeme.lemma.removesuffix("ion")

    # "sensation" → "sense" + "-ation"
    if lemma.endswith("at"):
        stem = find_lexeme(
            lemma=lemma.removesuffix("at") + "e",
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        )
        if stem:
            return Result(lexeme=lexeme, parts=[stem, Lexeme("L35048", "-ation")])

    # "manipulation" → "manipulate" + "-ion"
    stem = find_lexeme(
        lemma=lemma + "e",
        language=Language.ENGLISH,
        categories=[LexicalCategory.VERB],
    )
    if stem:
        return Result(lexeme=lexeme, parts=[stem, Lexeme("L35036", "-ion")])

    # "connection" → "connect" + "-ion"
    stem = find_lexeme(
        lemma=lemma,
        language=Language.ENGLISH,
        categories=[LexicalCategory.VERB],
    )
    return Result(lexeme=lexeme, parts=[stem, Lexeme("L35036", "-ion")])


@task(language=Language.ENGLISH, categories=[LexicalCategory.NOUN], include="......er$")
def en_er(lexeme: Lexeme) -> Result:
    lemma = lexeme.lemma.removesuffix("er")
    stem = None

    # "runner" → "run" + "-er"
    # "bidder" → "bid" + "-er"
    if lemma[-1] == lemma[-2]:
        stem = find_lexeme(
            lemma=lemma[:-1],
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        )

    # "killer" → "kill" + "-er"
    # "reader" → "read" + "-er"
    # "computer" → "compute" + "-er"
    if stem is None:
        stem = find_lexeme(
            lemma=lemma,
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        ) or find_lexeme(
            lemma=lemma + "e",
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        )

    return Result(
        lexeme=lexeme,
        parts=[stem, Lexeme("L29845", "-er")],
        types=[LexemeType.AGENT_NOUN],
    )


@task(language=Language.ENGLISH, categories=[LexicalCategory.NOUN], include="......or$")
def en_or(lexeme: Lexeme) -> Result:
    parts = [
        # "actor" → "act" + "-or"
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("or"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        )
        # "survivor" → "survive" + "-or"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("or") + "e",
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L29846", "-or"),
    ]
    return Result(lexeme=lexeme, parts=parts, types=[LexemeType.AGENT_NOUN])


@task(language=Language.ENGLISH, categories=[LexicalCategory.NOUN], include="......ee$")
def en_ee(lexeme: Lexeme) -> Result:
    parts = [
        # "examinee" → "examine" + "-ee"
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("e"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        )
        # "interviewee" → "interview" + "-ee"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("ee"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L47699", "-ee"),
    ]
    return Result(lexeme=lexeme, parts=parts, types=[LexemeType.AGENT_NOUN])


@task(
    language=Language.ENGLISH, categories=[LexicalCategory.VERB], include="......ize$"
)
def en_ize(lexeme: Lexeme) -> Result:
    parts = [
        # "colonize" → "colony" + "-ize"
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ize") + "y",
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        )
        # "pixelize" → "pixel" + "-ize"
        # "brutalize" → "brural" + "-ize"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("ize"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        )
        # "satirize" → "satire" + "-ize"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("ize") + "e",
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        ),
        Lexeme("L480567", "-ize"),
    ]
    return Result(lexeme=lexeme, parts=parts)


@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN],
    include=".....ment$",
)
def en_ment(lexeme: Lexeme) -> Result:
    parts = [
        # "acknowledgment" → "acknowledge" + "-ment"
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ment") + "e",
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        )
        # "abandonment" → "abandon" + "-ment"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("ment"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L29594", "-ment"),
    ]
    return Result(lexeme=lexeme, parts=parts)


@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN],
    include=".....ity$",
)
def en_ity(lexeme: Lexeme) -> Result:
    parts = [
        # "accountability" → "accountable" + "-ity"
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ity") + "e",
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ, LexicalCategory.NOUN],
        )
        # "absurdity" → "absurd" + "-ity"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("ity"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ, LexicalCategory.NOUN],
        ),
        Lexeme("L35038", "-ity"),
    ]
    return Result(lexeme=lexeme, parts=parts)


@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.VERB],
    include=".....ify$",
)
def en_ify(lexeme: Lexeme) -> Result:
    parts = [
        # "amplify" → "ample" + "-ify"
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ify") + "e",
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ, LexicalCategory.NOUN],
        )
        # "beastify" → "beast" + "-ify"
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("ify"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ, LexicalCategory.NOUN],
        ),
        Lexeme("L478506", "-ify"),
    ]
    return Result(lexeme=lexeme, parts=parts)


@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.ADJ],
    include="...-?like$",
)
def en_like(lexeme: Lexeme) -> Result:
    parts = [
        # "childlike" → "child" + "-like"
        # "snake-like" → "snake" + "-like"
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("like").removesuffix("-"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L615446", "-like"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "venomous" → "venom" + "-ous"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.ADJ],
    include="...ous$",
)
def en_ous(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ous"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L618508", "-ous"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "brotherhood" → "brother" + "-hood"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN],
    include="...hood$",
)
def en_hood(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("hood"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L252104", "-hood"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "friendship" → "friend" + "-ship"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN],
    include="...ship$",
)
def en_ship(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ship"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L348021", "-ship"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "freedom" → "free" + "-dom"
# "kingdom" → "king" + "-dom"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN],
    include="...dom$",
)
def en_dom(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("dom"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        ),
        Lexeme("L618510", "-dom"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "nonpayment" → "non-" + "payment"
# "nonaggressive" → "non-" + "aggressive"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
    include="^non-?......",
)
def en_non(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L15648", "non-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("non").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "supercomputer" → "super-" + "computer"
# "supercharge" → "super-" + "charge"
# "supernatural" → "super-" + "natural"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ, LexicalCategory.VERB],
    include="^super-?......",
)
def en_super(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L36094", "super-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("super").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "submarine" → "sub-" + "marine"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
    include="^sub-?......",
)
def en_sub(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L36093", "sub-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("sub").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "hypoactive" → "hypo-" + "active"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
    include="^hypo-?......",
)
def en_hypo(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L36100", "hypo-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("hypo").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "hypersonic" → "hyper-" + "sonic"
# "hypertext" → "hyper-" + "text"
@task(
    language=Language.ENGLISH,
    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
    include="^hyper-?......",
)
def en_hyper(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L36098", "hyper-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("hyper").removeprefix("-"),
            language=Language.ENGLISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "okänslig" → "o-" + "känslig"
@task(language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="^o...")
def sv_o(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L406921", "o-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("o"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.ADJ],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "målare" → "måla" + "-are"
@task(language=Language.SWEDISH, categories=[LexicalCategory.NOUN], include="...are$")
def sv_are(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("re"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L250345", "-are"),
    ]
    return Result(
        lexeme=lexeme,
        parts=parts,
        types=[LexemeType.VERBAL_NOUN, LexemeType.AGENT_NOUN],
    )


# "värdelös" → "värde" + "-lös"
@task(language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="...lös$")
def sv_los(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("lös"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L47685", "-lös"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "problemfri" → "problem" + "-fri"
@task(language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="...fri$")
def sv_fri(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("fri"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L47708", "-fri"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "rutinmässig" → "rutin" + "-mässig"
@task(language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="...mässig$")
def sv_massig(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("mässig"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L53569", "-mässig"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "hållbar" → "hålla" + "-bar"
@task(language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="...bar$")
def sv_bar(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("bar") + "a",
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L349047", "-bar"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "möjlighet" → "möjlig" + "-het"
@task(language=Language.SWEDISH, categories=[LexicalCategory.NOUN], include="...het$")
def sv_het(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("het"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.ADJ],
        ),
        Lexeme("L477760", "-het"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "motivera" → "motiv" + "-era"
@task(
    language=Language.SWEDISH,
    categories=[LexicalCategory.VERB],
    include="...era$",
    exclude="isera$",
)
def sv_era(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("era"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L590606", "-era"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "katalogisera" → "katalog" + "-isera"
# "globalisera" → "global" + "-isera"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="...isera$")
def sv_isera(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("isera"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        ),
        Lexeme("L590607", "-isera"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "överskatta" → "över-" + "skatta"
# "överambitiös" → "över-" + "ambitiös"
# "överkonsumtion" → "över-" + "konsumtion"
@task(
    language=Language.SWEDISH,
    categories=[LexicalCategory.VERB, LexicalCategory.ADJ, LexicalCategory.NOUN],
    include="^över...",
)
def sv_over(lexeme: Lexeme) -> Result:
    assert lexeme.category is not None
    parts = [
        Lexeme("L583836", "över-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("över"),
            language=Language.SWEDISH,
            categories=[lexeme.category],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "återresa" → "åter-" + "resa"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="^åter...")
def sv_ater(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L456508", "åter-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("åter"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "handling" → "handla" + "-ing"
@task(
    language=Language.SWEDISH,
    categories=[LexicalCategory.NOUN],
    include="...ing$",
    exclude="ning$",
)
def sv_ing(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ing") + "a",
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L591279", "-ing"),
    ]
    return Result(lexeme=lexeme, parts=parts, types=[LexemeType.VERBAL_NOUN])


# "tillverkning" → "tillverka" + "-ning"
@task(language=Language.SWEDISH, categories=[LexicalCategory.NOUN], include="...ning$")
def sv_ning(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ning") + "a",
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L230224", "-ning"),
    ]
    return Result(lexeme=lexeme, parts=parts, types=[LexemeType.VERBAL_NOUN])


# "avbryta" → "av-" + "bryta"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="^av...")
def sv_av(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L583405", "av-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("av"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "utandas" → "ut-" + "andas"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="^ut...")
def sv_ut(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L591605", "ut-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("ut"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "tillgå" → "till-" + "gå"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="^till...")
def sv_till(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L591609", "till-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("till"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "gitarrist" → "gitarr" + "-ist"
# "absurdist" → "absurd" + "-ist"
@task(language=Language.SWEDISH, categories=[LexicalCategory.NOUN], include="...ist$")
def sv_ist(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ist"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        ),
        Lexeme("L477925", "-ist"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "alkoholism" → "alkohol" + "-ism"
# "absurdism" → "absurd" + "-ism"
@task(language=Language.SWEDISH, categories=[LexicalCategory.NOUN], include="...ism$")
def sv_ism(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ism"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        ),
        Lexeme("L347287", "-ism"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "förkorta" → "för-" + "korta"
# "förenkla" → "för-" + "enkla"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="^för...")
def sv_for(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L347290", "för-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("för"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "föreställa" → "före-" + "ställa"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="^före...")
def sv_fore(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L583807", "före-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("före"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "omskapa" → "om-" + "skapa"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="^om...")
def sv_om(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L348192", "om-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("om"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "byggnad" → "bygga" + "-nad"
@task(language=Language.SWEDISH, categories=[LexicalCategory.NOUN], include="...nad$")
def sv_nad(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("nad") + "a",
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L593090", "-nad"),
    ]
    return Result(lexeme=lexeme, parts=parts, types=[LexemeType.VERBAL_NOUN])


@task(language=Language.SWEDISH, categories=[LexicalCategory.ADVERB], include="...vis$")
def sv_vis(lexeme: Lexeme) -> Result:
    # "delvis" → "del" + "-vis"
    # "dosvis" → "dos" + "-vis"
    # "punktvis" → "punkt" + "-vis"
    lemma = lexeme.lemma.removesuffix("vis")
    stem = find_lexeme(
        lemma=lemma,
        language=Language.SWEDISH,
        categories=[LexicalCategory.NOUN],
    )
    if stem:
        return Result(lexeme=lexeme, parts=[stem, Lexeme("L593311", "-vis")])

    # "kvartalsvis" → "kvartal" + "-s-" + "-vis"
    if lemma.endswith("s"):
        stem = find_lexeme(
            lemma=lemma.removesuffix("s"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        )
        return Result(
            lexeme=lexeme,
            parts=[
                stem,
                Lexeme("L54926", "-s-"),
                Lexeme("L593311", "-vis"),
            ],
        )

    # "möjligtvis" → "möjlig" + "-t" + "-vis"
    if lemma.endswith("t"):
        stem = find_lexeme(
            lemma=lemma.removesuffix("t"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.ADJ],
        )
        return Result(
            lexeme=lexeme,
            parts=[
                stem,
                Lexeme("L593310", "-t"),
                Lexeme("L593311", "-vis"),
            ],
        )

    return Result(lexeme=lexeme, parts=[])


# "snabbt" -> "snabb" + "-t"
@task(language=Language.SWEDISH, categories=[LexicalCategory.ADVERB], include="...t$")
def sv_t(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("t"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.ADJ],
        ),
        Lexeme("L593310", "-t"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "finskspråkig" → "finsk" + "-språkig"
@task(
    language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="...språkig$"
)
def sv_sprakig(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("språkig"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.ADJ],
        ),
        Lexeme("L593973", "-språkig"),
    ]
    return Result(lexeme=lexeme, parts=parts, types=[LexemeType.ABSOLUTE_ADJ])


# "angöra" → "an-" + "göra"
@task(language=Language.SWEDISH, categories=[LexicalCategory.VERB], include="^an...")
def sv_an(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L583404", "an-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("an"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


@task(language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="...ig$")
def sv_ig(lexeme: Lexeme) -> Result:
    if lexeme.lemma.endswith("lig"):
        parts = [
            # "liv" → "liv" + "-lig"
            find_lexeme(
                lemma=lexeme.lemma.removesuffix("lig"),
                language=Language.SWEDISH,
                categories=[LexicalCategory.NOUN],
            )
            # "lycklig" → "lycka" + "-lig"
            or find_lexeme(
                lemma=lexeme.lemma.removesuffix("lig") + "a",
                language=Language.SWEDISH,
                categories=[LexicalCategory.NOUN],
            ),
            Lexeme("L579134", "-lig"),
        ]
        return Result(lexeme=lexeme, parts=parts)

    # "kraftig" → "kraft" + "-ig"
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ig"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L579313", "-ig"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "dröm" → "dröm" + "-lik"
@task(language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="...lik$")
def sv_lik(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("lik"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        ),
        Lexeme("L615449", "-lik"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "våldsam" → "våld" + "-sam"
# "arbetsam" → "arbeta" + "-sam"
@task(language=Language.SWEDISH, categories=[LexicalCategory.ADJ], include="...sam$")
def sv_sam(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("sam"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        )
        or find_lexeme(
            lemma=lexeme.lemma.removesuffix("sam") + "a",
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L491120", "-sam"),
    ]
    return Result(lexeme=lexeme, parts=parts)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if args.task:
        execute_tasks = [task for task in tasks if task.name == args.task]
    else:
        execute_tasks = tasks

    if args.write:
        login_instance = create_login_instance()

    with History("prefix.pickle") as history:
        for task in execute_tasks:
            for result in task.execute(args.limit, history):
                lexeme = result.lexeme
                parts = result.parts
                summary = (
                    "combines "
                    + format_list(
                        [f"[[Lexeme:{part.qid}|{part.lemma}]]" for part in parts]
                    )
                    + " [[User:Kriobot#Task_2|#task2]]"
                )
                logging.info(f"[[Lexeme:{lexeme.qid}|{lexeme.lemma}]]) {summary}")
                if args.write:
                    combines = [
                        wbi_datatype.Lexeme(
                            value=part.qid,
                            prop_nr="P5238",
                            qualifiers=[
                                wbi_datatype.String(value=str(i + 1), prop_nr="P1545")
                            ],
                        )
                        for i, part in enumerate(parts)
                    ]
                    instance_of = [
                        wbi_datatype.ItemID(value=type.value, prop_nr="P31")
                        for type in result.types
                    ]
                    item = wbi_core.ItemEngine(
                        item_id=lexeme.qid, data=[*combines, *instance_of]
                    )
                    item.write(login_instance, edit_summary=summary)


if __name__ == "__main__":
    main()

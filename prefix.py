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


class Lexeme:
    def __init__(self, qid, lemma=None):
        self.qid = qid.removeprefix("http://www.wikidata.org/entity/")
        self.lemma = lemma

    def __str__(self):
        return f"Lexeme({self.qid}, {self.lemma})"


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
        category: LexicalCategory,
        transform: Callable[[str], Result],
        include=None,
        exclude=None,
        name=None,
    ):
        self.language = language
        self.category = category
        self.transform = transform
        self.include = include
        self.exclude = exclude
        self.name = name

    def _search_lexemes(self, limit: int):
        query = "SELECT ?lexeme ?lemma WHERE {\n"
        query += f"  ?lexeme dct:language wd:{self.language.value};\n"
        query += f"    wikibase:lexicalCategory wd:{self.category.value};\n"
        query += "    wikibase:lemma ?lemma.\n"

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
            lexemes.append(Lexeme(row["lexeme"]["value"], row["lemma"]["value"]))
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
    query = "SELECT ?lexeme WHERE {\n"
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
    return Lexeme(qid, lemma)


# "unbounded" → "un-" + "bounded"
@task(
    language=Language.ENGLISH,
    category=LexicalCategory.ADJ,
    include="^un.",
    exclude="^under",
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
@task(
    language=Language.ENGLISH,
    category=LexicalCategory.VERB,
    include="^un.",
    exclude="^under",
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


# "restless" → "rest" + "-less"
@task(language=Language.ENGLISH, category=LexicalCategory.ADJ, include=".less$")
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
@task(language=Language.ENGLISH, category=LexicalCategory.NOUN, include=".ness$")
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
@task(language=Language.ENGLISH, category=LexicalCategory.NOUN, include=".ist$")
def en_ist(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("ist"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
        ),
        Lexeme("L29847", "-ist"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "alcoholism" → "alcohol" + "-ism"
# "surrealism" → "surreal" + "-ism"
@task(language=Language.ENGLISH, category=LexicalCategory.NOUN, include=".ism$")
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
@task(language=Language.ENGLISH, category=LexicalCategory.ADJ, include=".ful$")
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
@task(language=Language.ENGLISH, category=LexicalCategory.NOUN, include=".ful$")
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


@task(language=Language.ENGLISH, category=LexicalCategory.ADJ, include=".able$")
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
@task(language=Language.ENGLISH, category=LexicalCategory.ADJ, include=".ly$")
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
    category=LexicalCategory.ADVERB,
    include=".ly$",
)
def en_ly_adverb(lexeme: Lexeme) -> Result:
    if lexeme.lemma.endswith("ally"):
        # "basically" → "basic" + "-ally"
        stem = find_lexeme(
            lemma=lexeme.lemma.removesuffix("ally"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ],
        )
        if stem:
            return Result(lexeme=lexeme, parts=[stem, Lexeme("L592202", "-ally")])

        # "mythically" → "mythical" + "-ly"
        stem = find_lexeme(
            lemma=lexeme.lemma.removesuffix("ly"),
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ],
        )
        return Result(lexeme=lexeme, parts=[stem, Lexeme("L28890", "-ly")])

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
            lemma=lexeme.lemma,
            language=Language.ENGLISH,
            categories=[LexicalCategory.ADJ],
        ),
        Lexeme("L28890", "-ly"),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "okänslig" → "o-" + "känslig"
@task(language=Language.SWEDISH, category=LexicalCategory.ADJ, include="^o.")
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
@task(language=Language.SWEDISH, category=LexicalCategory.NOUN, include=".are$")
def sv_are(lexeme: Lexeme) -> Result:
    parts = [
        find_lexeme(
            lemma=lexeme.lemma.removesuffix("re"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
        Lexeme("L250345", "-are"),
    ]
    return Result(lexeme=lexeme, parts=parts, types=[LexemeType.VERBAL_NOUN])


# "värdelös" → "värde" + "-lös"
@task(language=Language.SWEDISH, category=LexicalCategory.ADJ, include=".lös$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.ADJ, include=".fri$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.ADJ, include=".mässig$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.ADJ, include=".bar$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.NOUN, include=".het$")
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
    category=LexicalCategory.VERB,
    include=".era$",
    exclude=".isera$",
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
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include=".isera$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include="^över.")
def sv_over_verb(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L583836", "över-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("över"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.VERB],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "överambitiös" → "över-" + "ambitiös"
@task(language=Language.SWEDISH, category=LexicalCategory.ADJ, include="^över.")
def sv_over_adj(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L583836", "över-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("över"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.ADJ],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "överkonsumtion" → "över-" + "konsumtion"
@task(language=Language.SWEDISH, category=LexicalCategory.NOUN, include="^över.")
def sv_over_noun(lexeme: Lexeme) -> Result:
    parts = [
        Lexeme("L583836", "över-"),
        find_lexeme(
            lemma=lexeme.lemma.removeprefix("över"),
            language=Language.SWEDISH,
            categories=[LexicalCategory.NOUN],
        ),
    ]
    return Result(lexeme=lexeme, parts=parts)


# "återresa" → "åter-" + "resa"
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include="^åter.")
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
    category=LexicalCategory.NOUN,
    include=".ing$",
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
@task(language=Language.SWEDISH, category=LexicalCategory.NOUN, include=".ning$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include="^av.")
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
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include="^ut.")
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
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include="^till.")
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
@task(language=Language.SWEDISH, category=LexicalCategory.NOUN, include=".ist$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.NOUN, include=".ism$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include="^för.")
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
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include="^före.")
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
@task(language=Language.SWEDISH, category=LexicalCategory.VERB, include="^om.")
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
@task(language=Language.SWEDISH, category=LexicalCategory.NOUN, include=".nad$")
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


@task(language=Language.SWEDISH, category=LexicalCategory.ADVERB, include=".vis$")
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
@task(language=Language.SWEDISH, category=LexicalCategory.ADVERB, include=".t$")
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

# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: MIT

import logging
from enum import Enum
from wikibaseintegrator import wbi_core, wbi_datatype, wbi_functions
import sys
from datetime import datetime, timezone
import pickle
from pathlib import Path
import random
from typings import Callable, Optional

from common import create_login_instance

class Language(Enum):
    ENGLISH = 'Q1860'
    SWEDISH = 'Q9027'

class LexicalCategory(Enum):
    ADJ = 'Q34698'
    NOUN = 'Q1084'
    VERB = 'Q24905'

class Lexeme:
    def __init__(self, qid, lemma=None):
        self.qid = qid.removeprefix('http://www.wikidata.org/entity/')
        self.lemma = lemma

    def __str__(self):
        return f'Lexeme({self.qid}, {self.lemma})'

class History:
    '''
    Stores information about previously processed lexemes:
    - Successfully matched lexemes are skipped forever. This way errors can be
      corrected by humans and the bot won't try to reinsert incorrect
      information.
    - Unmatched lexemes will be processed again after some time to see if
      matching works this time. This reduces repeated queries.
    '''

    def __init__(self, filename: str):
        history_dir = Path('history')
        history_dir.mkdir(exist_ok=True)
        self.path = history_dir / filename
        try:
            with self.path.open('rb') as f:
                self.items = pickle.load(f)
        except FileNotFoundError:
            self.items = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self.path.open('wb') as f:
            pickle.dump(self.items, f)

    def add(self, lexeme: Lexeme, matched: bool, now: datetime = None):
        if now is None:
            now = datetime.now(timezone.utc)
        self.items[lexeme.qid] = (now, matched)

    def __contains__(self, lexeme) -> bool:
        if lexeme.qid not in self.items:
            return False
        last_checked, matched = self.items[lexeme.qid]
        # Matched don't expire.
        if matched:
            return True
        # Unmatched should expire in 1-2 weeks.
        now = datetime.now(timezone.utc)
        if (now - last_checked).days < 7:
            return True
        return random.random() > 1/7

class Task:
    def __init__(self, language: Language, category: LexicalCategory, transform: Callable[[str], list[Lexeme]], include=None, exclude=None):
        self.language = language
        self.category = category
        self.transform = transform
        self.include = include
        self.exclude = exclude

    def _search_lexemes(self, limit: int):
        '''Search lexemes matching the specified prefix or suffix.'''
        query =   'SELECT ?lexeme ?lemma WHERE {\n'
        query += f'  ?lexeme dct:language wd:{self.language.value};\n'
        query += f'    wikibase:lexicalCategory wd:{self.category.value};\n'
        query +=  '    wikibase:lemma ?lemma.\n'
        if self.include:
            query += f'  FILTER(REGEX(?lemma, "{self.include}"))\n'
        if self.exclude:
            query += f'  FILTER(!REGEX(?lemma, "{self.exclude}"))\n'
        # Ignore lexemes with existing combines (P5238) claims. These might be
        # previously added by this bot or humans.
        query += '  MINUS { ?lexeme wdt:P5238 []. }\n'
        query += '}\n'
        # Query extra lexemes to fill the limit because some lexemes may be
        # skipped later if no matching lexeme is found.
        query += f'LIMIT {10*limit}'
        data = wbi_functions.execute_sparql_query(query)
        lexemes = []
        for row in data['results']['bindings']:
            lexemes.append(Lexeme(row['lexeme']['value'], row['lemma']['value']))
        return lexemes

    def execute(self, limit: int, history: History):
        i = 0
        for lexeme in self._search_lexemes(limit):
            if i == limit:
                break
            if lexeme in history:
                continue
            parts = self.transform(lexeme.lemma)
            if all(parts):
                yield lexeme, parts
                i += 1
                history.add(lexeme, matched=True)
            else:
                history.add(lexeme, matched=False)

def find_lexeme(lemma: str, language: Language, categories: list[LexicalCategory]) -> Optional[Lexeme]:
    '''Search a single lexeme with the specified lemma.'''

    cats = ', '.join(f'wd:{cat.value}' for cat in categories)
    query =   'SELECT ?lexeme WHERE {\n'
    query += f'  ?lexeme dct:language wd:{language.value};\n'
    query += f'    wikibase:lexicalCategory ?category;\n'
    query +=  '    wikibase:lemma ?lemma.\n'
    query += f'  FILTER(?category IN ({cats}) && STR(?lemma) = "{lemma}")\n'
    query +=  '}\n'
    query +=  'LIMIT 2'

    data = wbi_functions.execute_sparql_query(query)
    results = data['results']['bindings']

    # To play it safe, let's continue only if we found a single lexeme.
    if len(results) != 1:
        return None

    qid = results[0]['lexeme']['value']
    return Lexeme(qid, lemma)

def main():
    logging.basicConfig(level=logging.INFO)

    tasks = [
        # "unbounded" → "un-" + "bounded"
        Task(
            language=Language.ENGLISH,
            category=LexicalCategory.ADJ,
            include='^un.',
            exclude='^under',
            transform=lambda lemma: (
                Lexeme('L15649', 'un-'),
                find_lexeme(
                    lemma=lemma.removeprefix('un'),
                    language=Language.ENGLISH,
                    categories=[LexicalCategory.ADJ],
                ),
            )
        ),
        # "unbox" → "un-" + "box"
        Task(
            language=Language.ENGLISH,
            category=LexicalCategory.VERB,
            include='^un.',
            exclude='^under',
            transform=lambda lemma: (
                Lexeme('L15649', 'un-'),
                find_lexeme(
                    lemma=lemma.removeprefix('un'),
                    language=Language.ENGLISH,
                    categories=[LexicalCategory.VERB],
                ),
            )
        ),
        # "restless" → "rest" + "-less"
        Task(
            language=Language.ENGLISH,
            category=LexicalCategory.ADJ,
            include='.less$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('less'),
                    language=Language.ENGLISH,
                    categories=[LexicalCategory.NOUN],
                ),
                Lexeme('L303186', '-less')
            )
        ),
        # "awkwardness" → "awkward" + "-ness"
        # "happiness" → "happy" + "-ness"
        Task(
            language=Language.ENGLISH,
            category=LexicalCategory.NOUN,
            include='.ness$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=(
                        lemma.removesuffix('iness')+'y' if lemma.endswith('iness')
                        else lemma.removesuffix('ness')
                    ),
                    language=Language.ENGLISH,
                    categories=[LexicalCategory.ADJ],
                ),
                Lexeme('L269834', '-ness'),
            )
        ),
        # "guitarist" → "guitar" + "-ist"
        # "surrealist" → "surreal" + "-ist"
        Task(
            language=Language.ENGLISH,
            category=LexicalCategory.NOUN,
            include='.ist$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('ist'),
                    language=Language.ENGLISH,
                    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
                ),
                Lexeme('L29847', '-ist'),
            ),
        ),
        # "alcoholism" → "alcohol" + "-ism"
        # "surrealism" → "surreal" + "-ism"
        Task(
            language=Language.ENGLISH,
            category=LexicalCategory.NOUN,
            include='.ism$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('ism'),
                    language=Language.ENGLISH,
                    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
                ),
                Lexeme('L29596', '-ism'),
            ),
        ),
        # "okänslig" → "o-" + "känslig"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.ADJ,
            include='^o.',
            transform=lambda lemma: (
                Lexeme('L406921', 'o-'),
                find_lexeme(
                    lemma=lemma.removeprefix('o'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.ADJ],
                ),
            )
        ),
        # "målare" → "måla" + "-are"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.NOUN,
            include='.are$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('re'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
                Lexeme('L250345', '-are'),
            ),
        ),
        # "värdelös" → "värde" + "-lös"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.ADJ,
            include='.lös$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('lös'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.NOUN],
                ),
                Lexeme('L47685', '-lös'),
            ),
        ),
        # "problemfri" → "problem" + "-fri"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.ADJ,
            include='.fri$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('fri'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.NOUN],
                ),
                Lexeme('L47708', '-fri'),
            ),
        ),
        # "rutinmässig" → "rutin" + "-mässig"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.ADJ,
            include='.mässig$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('mässig'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.NOUN],
                ),
                Lexeme('L53569', '-mässig'),
            ),
        ),
        # "hållbar" → "hålla" + "-bar"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.ADJ,
            include='.bar$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('bar')+'a',
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
                Lexeme('L349047', '-bar'),
            ),
        ),
        # "möjlighet" → "möjlig" + "-het"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.NOUN,
            include='.het$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('het'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.ADJ],
                ),
                Lexeme('L477760', '-het'),
            ),
        ),
        # "motivera" → "motiv" + "-era"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.VERB,
            include='.era$',
            exclude='.isera$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('era'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.NOUN],
                ),
                Lexeme('L590606', '-era'),
            ),
        ),
        # "katalogisera" → "katalog" + "-isera"
        # "globalisera" → "global" + "-isera"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.VERB,
            include='.isera$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('isera'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
                ),
                Lexeme('L590607', '-isera'),
            ),
        ),
        # "överskatta" → "över-" + "skatta"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.VERB,
            include='^över.',
            transform=lambda lemma: (
                Lexeme('L583836', 'över-'),
                find_lexeme(
                    lemma=lemma.removeprefix('över'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
            ),
        ),
        # "överambitiös" → "över-" + "ambitiös"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.ADJ,
            include='^över.',
            transform=lambda lemma: (
                Lexeme('L583836', 'över-'),
                find_lexeme(
                    lemma=lemma.removeprefix('över'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.ADJ],
                ),
            ),
        ),
        # "överkonsumtion" → "över-" + "konsumtion"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.NOUN,
            include='^över.',
            transform=lambda lemma: (
                Lexeme('L583836', 'över-'),
                find_lexeme(
                    lemma=lemma.removeprefix('över'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.NOUN],
                ),
            ),
        ),
        # "återresa" → "åter-" + "resa"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.VERB,
            include='^åter.',
            transform=lambda lemma: (
                Lexeme('L456508', 'åter-'),
                find_lexeme(
                    lemma=lemma.removeprefix('åter'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
            ),
        ),
        # "handling" → "handla" + "-ing"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.NOUN,
            include='.ing$',
            exclude='ning$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('ing')+'a',
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
                Lexeme('L591279', '-ing'),
            ),
        ),
        # "tillverkning" → "tillverka" + "-ning"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.NOUN,
            include='.ning$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('ning')+'a',
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
                Lexeme('L230224', '-ning'),
            ),
        ),
        # "avbryta" → "av-" + "bryta"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.VERB,
            include='^av.',
            transform=lambda lemma: (
                Lexeme('L583405', 'av-'),
                find_lexeme(
                    lemma=lemma.removeprefix('av'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
            ),
        ),
        # "utandas" → "ut-" + "andas"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.VERB,
            include='^ut.',
            transform=lambda lemma: (
                Lexeme('L591605', 'ut-'),
                find_lexeme(
                    lemma=lemma.removeprefix('ut'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
            ),
        ),
        # "tillgå" → "till-" + "gå"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.VERB,
            include='^till.',
            transform=lambda lemma: (
                Lexeme('L591609', 'till-'),
                find_lexeme(
                    lemma=lemma.removeprefix('till'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
            ),
        ),
        # "gitarrist" → "gitarr" + "-ist"
        # "absurdist" → "absurd" + "-ist"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.NOUN,
            include='.ist$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('ist'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
                ),
                Lexeme('L477925', '-ist'),
            ),
        ),
        # "alkoholism" → "alkohol" + "-ism"
        # "absurdism" → "absurd" + "-ism"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.NOUN,
            include='.ism$',
            transform=lambda lemma: (
                find_lexeme(
                    lemma=lemma.removesuffix('ism'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.NOUN, LexicalCategory.ADJ],
                ),
                Lexeme('L347287', '-ism'),
            ),
        ),
        # "förkorta" → "för-" + "korta"
        Task(
            language=Language.SWEDISH,
            category=LexicalCategory.VERB,
            include='^för.',
            transform=lambda lemma: (
                Lexeme('L347290', 'för-'),
                find_lexeme(
                    lemma=lemma.removeprefix('för'),
                    language=Language.SWEDISH,
                    categories=[LexicalCategory.VERB],
                ),
            ),
        ),
    ]

    write = '--write' in sys.argv
    limit = 50 if write else 5

    if write:
        login_instance = create_login_instance()

    with History('prefix.pickle') as history:
        for task in tasks:
            for lexeme, parts in task.execute(limit, history):
                assert len(parts) == 2
                logging.info(f'"{lexeme.lemma}" ({lexeme.qid}) combines "{parts[0].lemma}" ({parts[0].qid}) and "{parts[1].lemma}" ({parts[1].qid})')
                if write:
                    summary = f'combines [[Lexeme:{parts[0].qid}|{parts[0].lemma}]] and [[Lexeme:{parts[1].qid}|{parts[1].lemma}]] [[User:Kriobot#Task_2|#task2]]'
                    data = [wbi_datatype.Lexeme(value=part.qid, prop_nr='P5238', qualifiers=[wbi_datatype.String(value=str(i+1), prop_nr='P1545')]) for i, part in enumerate(parts)]
                    item = wbi_core.ItemEngine(item_id=lexeme.qid, data=data)
                    item.write(login_instance, edit_summary=summary)

if __name__ == '__main__':
    main()

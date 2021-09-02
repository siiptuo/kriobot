import logging
from enum import Enum
from wikibaseintegrator import wbi_core, wbi_datatype, wbi_functions

from common import create_login_instance

class Language(Enum):
    ENGLISH = 'Q1860'
    SWEDISH = 'Q9027'

class LexicalCategory(Enum):
    ADJ = 'Q34698'
    NOUN = 'Q1084'
    VERB = 'Q24905'

class Lexeme:
    def __init__(self, qid, lemma):
        self.qid = qid.removeprefix('http://www.wikidata.org/entity/')
        self.lemma = lemma

    def __str__(self):
        return f'Lexeme({self.qid}, {self.lemma})'

class Task:
    def __init__(self, language: Language, from_category: LexicalCategory, to_category: LexicalCategory, prefix=None, suffix=None):
        if prefix and suffix:
            raise Exception('Having both prefix and suffix is not supported.')
        self.language = language
        self.from_category = from_category
        self.to_category = to_category
        self.prefix = prefix
        self.suffix = suffix

    def _search_lexemes(self):
        '''Search lexemes matching the specified prefix or suffix.'''
        query = 'SELECT ?lexeme ?lemma WHERE {'
        query += f' ?lexeme dct:language wd:{self.language.value};'
        query += f' wikibase:lexicalCategory wd:{self.from_category.value};'
        query += ' wikibase:lemma ?lemma.'
        if self.prefix:
            query += f' FILTER(REGEX(?lemma, "^{self.prefix.lemma[:-1]}"))'
        elif self.suffix:
            query += f' FILTER(REGEX(?lemma, "{self.suffix.lemma[1:]}$"))'
        # Ignore lexemes with existing combines (P5238) claims. These might be
        # previously added by this bot or humans.
        query += ' MINUS { ?lexeme wdt:P5238 []. }'
        query += '}'
        query += ' LIMIT 5'
        data = wbi_functions.execute_sparql_query(query)
        lexemes = []
        for row in data['results']['bindings']:
            lexemes.append(Lexeme(row['lexeme']['value'], row['lemma']['value']))
        return lexemes

    def _query_lexeme(self, lemma):
        '''Search a single lexeme with the specified lemma.'''
        query =   'SELECT ?lexeme WHERE {\n'
        query += f'  ?lexeme dct:language wd:{self.language.value};\n'
        query += f'          wikibase:lexicalCategory wd:{self.to_category.value};\n'
        query +=  '          wikibase:lemma ?lemma.\n'
        query += f'  FILTER(STR(?lemma) = "{lemma}")\n'
        query +=  '}'
        data = wbi_functions.execute_sparql_query(query)
        # To play it safe, let's continue only if we found a single lexeme.
        if len(data['results']['bindings']) != 1:
            return None
        return Lexeme(data['results']['bindings'][0]['lexeme']['value'], lemma)

    def execute(self):
        for lexeme in self._search_lexemes():
            if self.prefix:
                stem = lexeme.lemma[len(self.prefix.lemma)-1:]
                if match := self._query_lexeme(stem):
                    yield lexeme, [self.prefix, match]
            elif self.suffix:
                stem = lexeme.lemma[:-len(self.suffix.lemma)+1]
                if match := self._query_lexeme(stem):
                    yield lexeme, [match, self.suffix]

def main():
    logging.basicConfig(level=logging.INFO)

    tasks = [
        Task(
            language=Language.ENGLISH,
            from_category=LexicalCategory.ADJ,
            to_category=LexicalCategory.ADJ,
            prefix=Lexeme('L15649', 'un-')
        ),
        Task(
            language=Language.ENGLISH,
            from_category=LexicalCategory.ADJ,
            to_category=LexicalCategory.NOUN,
            suffix=Lexeme('L303186', '-less')
        ),
        Task(
            language=Language.ENGLISH,
            from_category=LexicalCategory.NOUN,
            to_category=LexicalCategory.ADJ,
            suffix=Lexeme('L269834', '-ness')
        ),
        Task(
            language=Language.SWEDISH,
            from_category=LexicalCategory.ADJ,
            to_category=LexicalCategory.ADJ,
            prefix=Lexeme('L406921', 'o-')
        ),
    ]

    login_instance = create_login_instance()

    for task in tasks:
        for lexeme, parts in task.execute():
            assert len(parts) == 2
            logging.info(f'"{lexeme.lemma}" ({lexeme.qid}) combines "{parts[0].lemma}" ({parts[0].qid}) and "{parts[1].lemma}" ({parts[1].qid})')
            summary = f'combines [[Lexeme:{parts[0].qid}|{parts[0].lemma}]] and [[Lexeme:{parts[1].qid}|{parts[1].lemma}]] [[User:Kriobot#Task_2|#task2]]'
            data = [wbi_datatype.Lexeme(value=part.qid, prop_nr='P5238', qualifiers=[wbi_datatype.String(value=str(i+1), prop_nr='P1545')]) for i, part in enumerate(parts)]
            item = wbi_core.ItemEngine(item_id=lexeme.qid, data=data)
            item.write(login_instance, edit_summary=summary)

if __name__ == '__main__':
    main()

# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: MIT

import csv
import json
import logging
import os
import urllib.request
from collections import defaultdict
from wikibaseintegrator import wbi_core, wbi_login, wbi_datatype, wbi_functions

def classify(conjugations):
    '''
    >>> classify(('kalla', 'kallar', 'kallade', 'kallat'))
    ('kall', '1')
    >>> classify(('stänga', 'stänger', 'stängde', 'stängt'))
    ('stäng', '2a')
    >>> classify(('tända', 'tänder', 'tände', 'tänt'))
    ('tänd', '2a')
    >>> classify(('läsa', 'läser', 'läste', 'läst'))
    ('läs', '2b')
    >>> classify(('sy', 'syr', 'sydde', 'sytt'))
    ('sy', '3')
    >>> classify(('stryka', 'skryker', 'strök', 'strukit'))
    (None, '4')
    >>> classify(('göra', 'gör', 'gjorde', 'gjort'))
    (None, '4')
    '''
    if all(c.endswith(suffix) for c, suffix in zip(conjugations, ('a', 'ar', 'ade', 'at'))):
        return (conjugations[0][:-1], '1')
    if all(c.endswith(suffix) for c, suffix in zip(conjugations, ('a', 'er', 'de', 't'))):
        return (conjugations[0][:-1], '2a')
    if all(c.endswith(suffix) for c, suffix in zip(conjugations, ('a', 'er', 'te', 't'))):
        return (conjugations[0][:-1], '2b')
    if all(c.endswith(suffix) for c, suffix in zip(conjugations, ('', 'r', 'dde', 'tt'))):
        return (conjugations[0], '3')
    return (None, '4')

CONJUGATION_GROUP_IDS = {'1': 'Q106617269', '2a': 'Q106617270', '2b': 'Q106617271', '3': 'Q106617272', '4': 'Q106617274'}

ACTIVE_VOICE_ID = 'Q1317831'
INFINITIVE_ID = 'Q179230'
PRESENT_TENSE_ID = 'Q192613'
PRETERITE_ID = 'Q442485'
SUPINE_ID = 'Q548470'

def main():
    logging.basicConfig(level=logging.INFO)

    login_instance = wbi_login.Login(user=os.environ['WIKIDATA_USERNAME'], pwd=os.environ['WIKIDATA_PASSWORD'])

    lexemes = defaultdict(lambda: [None, None, None, None])
    data = wbi_functions.execute_sparql_query(
        '''
        SELECT ?lexeme (GROUP_CONCAT(?feature; SEPARATOR = ",") AS ?features) (SAMPLE(?text) AS ?conjugation) WHERE {
          ?lexeme dct:language wd:Q9027;
            wikibase:lexicalCategory wd:Q24905;
            ontolex:lexicalForm ?form.
          ?form wikibase:grammaticalFeature ?feature;
            ontolex:representation ?text.
          MINUS { ?lexeme wdt:P5186 [] }
        }
        GROUP BY ?lexeme ?form
        '''
    )
    for row in data['results']['bindings']:
        lexeme = row['lexeme']['value'].removeprefix('http://www.wikidata.org/entity/')
        features = [feature.removeprefix('http://www.wikidata.org/entity/') for feature in row['features']['value'].split(',')]
        conjugation = row['conjugation']['value']
        if ACTIVE_VOICE_ID in features:
            if INFINITIVE_ID in features:
                lexemes[lexeme][0] = conjugation
            elif PRESENT_TENSE_ID in features:
                lexemes[lexeme][1] = conjugation
            elif PRETERITE_ID in features:
                lexemes[lexeme][2] = conjugation
            elif SUPINE_ID in features:
                lexemes[lexeme][3] = conjugation

    i = 0
    for lexeme, conjugations in lexemes.items():
        if i > 50:
            break
        if all(conjugations):
            stem, klass = classify(conjugations)
            logging.info(f'lexeme={lexeme} stem={stem} class={klass}')

            data = [wbi_datatype.ItemID(value=CONJUGATION_GROUP_IDS[klass], prop_nr='P5186')]
            summary = 'add conjugation class'
            if stem:
                data.append(wbi_datatype.MonolingualText(text=stem, prop_nr='P5187', language='sv'))
                summary += ' and stem'
            summary += ' [[User:Kriobot#Task_1|#task1]]'

            item = wbi_core.ItemEngine(item_id=lexeme, data=data)
            item.write(login_instance, edit_summary=summary)

            i += 1

if __name__ == '__main__':
    main()

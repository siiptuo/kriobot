# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: MIT

import csv
import json
import logging
import urllib.request
from collections import defaultdict
from wikibaseintegrator import wbi_core, wbi_datatype, wbi_functions

from common import create_login_instance

VOWELS = set('aeiouyåäö')

def conjugate1(infinitive):
    '''
    >>> conjugate1('kalla')
    ('kall', ('kalla', 'kallar', 'kallade', 'kallat'))
    '''
    stem = infinitive[:-1]
    present = stem + 'ar'
    preterite = stem + 'ade'
    supine = stem + 'at'
    return stem, (infinitive, present, preterite, supine)

def conjugate2a(infinitive):
    '''
    >>> conjugate2a('stänga')
    ('stäng', ('stänga', 'stänger', 'stängde', 'stängt'))
    >>> conjugate2a('tända')
    ('tänd', ('tända', 'tänder', 'tände', 'tänt'))
    >>> conjugate2a('röra')
    ('rör', ('röra', 'rör', 'rörde', 'rört'))
    >>> conjugate2a('lyda')
    ('lyd', ('lyda', 'lyder', 'lydde', 'lytt'))
    >>> conjugate2a('träda')
    ('träd', ('träda', 'träder', 'trädde', 'trätt'))
    '''
    if infinitive.endswith('ra'):
        stem = infinitive[:-1]
        present = stem
    else:
        stem = infinitive[:-1]
        present = stem + 'er'
    if stem.endswith('d') and stem[-2] not in VOWELS:
        preterite = stem + 'e'
    else:
        preterite = stem + 'de'
    if stem.endswith('d'):
        supine = stem[:-1] + ('tt' if stem[-2] in VOWELS else 't')
    else:
        supine = stem + 't'
    return stem, (infinitive, present, preterite, supine)

def conjugate2b(infinitive):
    '''
    >>> conjugate2b('läsa')
    ('läs', ('läsa', 'läser', 'läste', 'läst'))
    >>> conjugate2b('gifta')
    ('gift', ('gifta', 'gifter', 'gifte', 'gift'))
    '''
    stem = infinitive[:-1]
    present = stem + 'er'
    if stem.endswith('t'):
        preterite = stem + 'e'
        supine = stem
    else:
        preterite = stem + 'te'
        supine = stem + 't'
    return stem, (infinitive, present, preterite, supine)

def conjugate3(infinitive):
    '''
    >>> conjugate3('sy')
    ('sy', ('sy', 'syr', 'sydde', 'sytt'))
    '''
    stem = infinitive
    present = stem + 'r'
    preterite = stem + 'dde'
    supine = stem + 'tt'
    return stem, (infinitive, present, preterite, supine)

def conjugate(infinitive, group):
    if group == '1':
        return conjugate1(infinitive)
    if group == '2a':
        return conjugate2a(infinitive)
    if group == '2b':
        return conjugate2b(infinitive)
    if group == '3':
        return conjugate3(infinitive)
    raise Exception(f'unsupported group: {group}')

def classify(conjugations):
    '''
    >>> classify(('kalla', 'kallar', 'kallade', 'kallat'))
    ('kall', '1')
    >>> classify(('stänga', 'stänger', 'stängde', 'stängt'))
    ('stäng', '2a')
    >>> classify(('tända', 'tänder', 'tände', 'tänt'))
    ('tänd', '2a')
    >>> classify(('röra', 'rör', 'rörde', 'rört'))
    ('rör', '2a')
    >>> classify(('läsa', 'läser', 'läste', 'läst'))
    ('läs', '2b')
    >>> classify(('sy', 'syr', 'sydde', 'sytt'))
    ('sy', '3')
    >>> classify(('stryka', 'skryker', 'strök', 'strukit'))
    (None, '4')
    >>> classify(('göra', 'gör', 'gjorde', 'gjort'))
    (None, '4')
    >>> classify(('ha', 'har', 'hade', 'haft'))
    (None, '4')
    '''
    for group in ('1', '2a', '2b', '3'):
        stem, candidates = conjugate(conjugations[0], group)
        if candidates == conjugations:
            return stem, group
    return (None, '4')

CONJUGATION_GROUP_IDS = {'1': 'Q106617269', '2a': 'Q106617270', '2b': 'Q106617271', '3': 'Q106617272', '4': 'Q106617274'}

ACTIVE_VOICE_ID = 'Q1317831'
INFINITIVE_ID = 'Q179230'
PRESENT_TENSE_ID = 'Q192613'
PRETERITE_ID = 'Q442485'
SUPINE_ID = 'Q548470'

def main():
    logging.basicConfig(level=logging.INFO)

    login_instance = create_login_instance()

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
        LIMIT 100
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

    for lexeme, conjugations in lexemes.items():
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

if __name__ == '__main__':
    main()

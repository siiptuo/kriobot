# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: MIT

import logging
from wikibaseintegrator import wbi_core, wbi_login, wbi_datatype, wbi_functions

FIFTH_DECLENSION_EXCEPTIONS = [('mus', 'möss'), ('gås', 'gäss'), ('man', 'män')]

def classify(singular, plural):
    '''
    >>> classify('flicka', 'flickor')
    1
    >>> classify('våg', 'vågor')
    1
    >>> classify('ros', 'rosor')
    1
    >>> classify('finger', 'fingrar')
    2
    >>> classify('arm', 'armar')
    2
    >>> classify('hund', 'hundar')
    2
    >>> classify('sjö', 'sjöar')
    2
    >>> classify('pojke', 'pojkar')
    2
    >>> classify('sjukdom', 'sjukdomar')
    2
    >>> classify('främling', 'främlingar')
    2
    >>> classify('afton', 'aftnar')
    2
    >>> classify('sommar', 'somrar')
    2
    >>> classify('moder', 'mödrar')
    2
    >>> classify('mor', 'mödrar')
    2
    >>> classify('park', 'parker')
    3
    >>> classify('museum', 'museer')
    3
    >>> classify('sko', 'skor')
    3
    >>> classify('fiende', 'fiender')
    3
    >>> classify('hand', 'händer')
    3
    >>> classify('land', 'länder')
    3
    >>> classify('bok', 'böcker')
    3
    >>> classify('nöt', 'nötter')
    3
    >>> classify('bi', 'bin')
    4
    >>> classify('äpple', 'äpplen')
    4
    >>> classify('öga', 'ögon')
    4
    >>> classify('öra', 'öron')
    4
    >>> classify('barn', 'barn')
    5
    >>> classify('djur', 'djur')
    5
    >>> classify('lärare', 'lärare')
    5
    >>> classify('mus', 'möss')
    5
    >>> classify('gås', 'gäss')
    5
    >>> classify('man', 'män')
    5
    '''
    if singular == plural:
        return 5
    for s, p in FIFTH_DECLENSION_EXCEPTIONS:
        if singular.endswith(s) and plural.endswith(p):
            return 5
    if not singular.endswith('o') and plural.endswith('or'):
        return 1
    if not singular.endswith('a') and plural.endswith('ar'):
        return 2
    if plural.endswith('r'):
        return 3
    if plural.endswith('n'):
        return 4
    return None

DECLENSION_ID = {1: 'Q106602496', 2: 'Q106602498', 3: 'Q106602499', 4: 'Q106602501', 5: 'Q106602503'}

def main():
    logging.basicConfig(level=logging.INFO)

    login_instance = wbi_login.Login(user=os.environ['WIKIDATA_USERNAME'], pwd=os.environ['WIKIDATA_PASSWORD'])

    data = wbi_functions.execute_sparql_query(
        '''
        SELECT ?lexeme ?singular ?plural WHERE {
          ?lexeme dct:language wd:Q9027;
            wikibase:lexicalCategory wd:Q1084;
            wikibase:lemma ?singular;
            ontolex:lexicalForm ?form.
          ?form wikibase:grammaticalFeature wd:Q131105, wd:Q146786, wd:Q53997857;
            ontolex:representation ?plural.
          FILTER(NOT EXISTS { ?lexeme wdt:P5911 []. })
        }
        '''
    )

    i = 0
    for row in data['results']['bindings']:
        if i > 50:
            break

        lexeme = row['lexeme']['value'].removeprefix('http://www.wikidata.org/entity/')
        singular = row['singular']['value']
        plural = row['plural']['value']
        klass = classify(singular, plural)
        logging.info(f'lexeme={lexeme} singular={singular} plural={plural} class={klass}')
        if not klass:
            continue

        data = [wbi_datatype.ItemID(value=DECLENSION_ID[klass], prop_nr='P5911')]
        item = wbi_core.ItemEngine(item_id=lexeme, data=data)
        item.write(login_instance, edit_summary='add declension')

        i += 1

if __name__ == '__main__':
    main()

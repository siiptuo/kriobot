import logging
from wikibaseintegrator import wbi_core, wbi_datatype, wbi_functions

from common import create_login_instance

def query_lexeme(word):
    data = wbi_functions.execute_sparql_query(
        f'''
        SELECT ?lexeme WHERE {{
          ?lexeme dct:language wd:Q1860;
            wikibase:lexicalCategory wd:Q34698;
            wikibase:lemma "{word}"@en.
        }}
        '''
    )
    if len(data['results']['bindings']) == 0 or len(data['results']['bindings']) > 1:
        return None
    return data['results']['bindings'][0]['lexeme']['value'].removeprefix('http://www.wikidata.org/entity/')

def main():
    logging.basicConfig(level=logging.INFO)

    login_instance = create_login_instance()

    data = wbi_functions.execute_sparql_query(
        '''
        SELECT ?lexeme ?lemma WHERE {
          ?lexeme dct:language wd:Q1860;
            wikibase:lexicalCategory wd:Q34698;
            wikibase:lemma ?lemma.
          FILTER(REGEX(?lemma, "^un.*") && !REGEX(?lemma, "^under.*"))
          MINUS { ?lexeme wdt:P5238 []. }
        }
        LIMIT 10
        '''
    )

    for row in data['results']['bindings']:
        lexeme_id = row['lexeme']['value'].removeprefix('http://www.wikidata.org/entity/')
        lemma = row['lemma']['value']
        part_id = query_lexeme(lemma.removeprefix('un'))
        logging.info(f'lexeme={lexeme_id} lemma={lemma} part={part_id}')
        if part_id:
            data = [wbi_datatype.Lexeme(value='L15649', prop_nr='P5238', qualifiers=[wbi_datatype.String(value="1", prop_nr='P1545')]),
                    wbi_datatype.Lexeme(value=part_id, prop_nr='P5238', qualifiers=[wbi_datatype.String(value="2", prop_nr='P1545')])]
            item = wbi_core.ItemEngine(item_id=lexeme_id, data=data)
            item.write(login_instance, edit_summary="combines \"un-\" prefix [[User:Kriobot#Task_2|#task2]]")

if __name__ == '__main__':
    main()

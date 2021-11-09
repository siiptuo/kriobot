# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: MIT

import csv
import random
from dataclasses import dataclass

from wikibaseintegrator import wbi_core, wbi_datatype, wbi_functions

from common import create_login_instance


@dataclass
class Lexeme:
    qid: str
    lemma: str
    category: str


def find_parts(lexemes, lexeme, lemma, parts):
    if lemma == "":
        return [parts]
    output = []
    for candidate in lexemes.values():
        if candidate.qid == lexeme.qid:
            continue
        if candidate.lemma.startswith("-") and candidate.lemma.endswith("-"):
            if (
                len(parts) > 0
                and lemma.startswith(candidate.lemma[1:-1])
                and candidate.lemma[1:-1] != lemma
            ):
                output.extend(
                    find_parts(
                        lexemes,
                        lexeme,
                        lemma[len(candidate.lemma) - 2 :],
                        [*parts, candidate],
                    )
                )
        elif candidate.lemma.startswith("-"):
            if len(parts) > 0 and candidate.lemma[1:] == lemma:
                output.append([*parts, candidate])
        elif candidate.lemma.endswith("-"):
            if len(parts) == 0 and lemma.startswith(candidate.lemma[:-1]):
                output.extend(
                    find_parts(
                        lexemes,
                        lexeme,
                        lemma[len(candidate.lemma) - 1 :],
                        [*parts, candidate],
                    )
                )
        else:
            if lexeme.category != candidate.category:
                continue
            if len(candidate.lemma) < 3:
                continue
            if not lemma.startswith(candidate.lemma):
                continue
            output.extend(
                find_parts(
                    lexemes, lexeme, lemma[len(candidate.lemma) :], [*parts, candidate]
                )
            )
    return output


def format_list(items):
    """
    >>> format_list(['A', 'B'])
    'A and B'
    >>> format_list(['A', 'B', 'C'])
    'A, B and C'
    """
    assert len(items) >= 2
    return ", ".join(items[:-1]) + " and " + items[-1]


def write_lexeme(login_instance, lexeme, parts):
    summary = "combines " + format_list(
        [f"[[Lexeme:{part.qid}|{part.lemma}]]" for part in parts]
    )
    print(summary)
    data = [
        wbi_datatype.Lexeme(
            value=part.qid,
            prop_nr="P5238",
            qualifiers=[wbi_datatype.String(value=str(i + 1), prop_nr="P1545")],
        )
        for i, part in enumerate(parts)
    ]
    item = wbi_core.ItemEngine(item_id=lexeme.qid, data=data)
    item.write(login_instance, edit_summary=summary)


def main():
    login_instance = create_login_instance()

    lexemes = {}

    with open("data/lexemes-sv.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["lang"] == "sv":
                lexemes[row["lexeme"]] = Lexeme(
                    qid=row["lexeme"].removeprefix("http://www.wikidata.org/entity/"),
                    lemma=row["lemma"],
                    category=row["lexicalCategory"].removeprefix(
                        "http://www.wikidata.org/entity/"
                    ),
                )

    with open("data/combines-sv.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lexeme = row["lexeme"].removeprefix("http://www.wikidata.org/entity/")
            if lexeme in lexemes:
                del lexemes[lexeme]

    shuffled = list(lexemes.values())
    random.shuffle(shuffled)

    for lexeme in shuffled:
        if parts := find_parts(lexemes, lexeme, lexeme.lemma, []):
            print(
                f"lemma: {lexeme.lemma} (http://www.wikidata.org/entity/{lexeme.qid})"
            )
            for i, part in enumerate(parts):
                s = '"' + '" + "'.join(p.lemma for p in part) + '"'
                print(f"{i+1}. {s}")
            option = input("select: ")
            if option:
                i = int(option) - 1
                write_lexeme(login_instance, lexeme, parts[i])
            print()


if __name__ == "__main__":
    main()

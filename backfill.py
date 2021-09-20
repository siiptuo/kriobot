# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: MIT

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from time import sleep

from prefix import History, Lexeme


class MediaWikiError(Exception):
    pass


def do_request(params):
    query = urllib.parse.urlencode(
        {**params, "format": "json", "errorformat": "plaintext", "formatversion": 2}
    )
    url = "https://wikidata.org/w/api.php?" + query
    with urllib.request.urlopen(url) as res:
        if res.status != 200:
            raise MediaWikiError(res.reason)
        body = json.load(res)
        if "errors" in body:
            raise MediaWikiError(". ".join(error["text"] for error in body["errors"]))
        return body


def usercontribs(username, kontinue=None):
    print("usercontribs", username, "with continue", kontinue, file=sys.stderr)
    params = {
        "action": "query",
        "list": "usercontribs",
        "ucuser": username,
        "uclimit": "max",
    }
    if kontinue is not None:
        params.update(kontinue)
    response = do_request(params)
    yield from response["query"]["usercontribs"]
    if "continue" in response:
        sleep(5)
        yield from usercontribs(username, response["continue"])


def main():
    with History("prefix.pickle") as history:
        for contrib in usercontribs("Kriobot"):
            if not contrib["title"].startswith("Lexeme:"):
                continue
            if "#task2" not in contrib["comment"]:
                continue
            lexeme = Lexeme(contrib["title"].removeprefix("Lexeme:"))
            timestamp = datetime.strptime(contrib["timestamp"], "%Y-%m-%dT%H:%M:%S%z")
            history.add(lexeme, matched=True, now=timestamp)


if __name__ == "__main__":
    main()

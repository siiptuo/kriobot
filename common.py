# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: MIT

import os
from wikibaseintegrator import wbi_login
from wikibaseintegrator.wbi_config import config as wbi_config

__version__ = '1.0'

wbi_config['USER_AGENT_DEFAULT'] = f'Kriobot/{__version__} (https://www.wikidata.org/wiki/User:Kriobot)'

def create_login_instance():
    login_instance = wbi_login.Login(user=os.environ['WIKIDATA_USERNAME'], pwd=os.environ['WIKIDATA_PASSWORD'])

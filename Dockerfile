# SPDX-FileCopyrightText: 2021 Tuomas Siipola
# SPDX-License-Identifier: CC0-1.0

FROM python:3.9.6-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]

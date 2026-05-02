#!/usr/bin/env bash
set -e
apt-get update -y
apt-get install -y tesseract-ocr poppler-utils libgl1-mesa-glx
python3 -m pip install --upgrade pip
python3 -m pip install -r backend/requirements.txt
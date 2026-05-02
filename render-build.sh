#!/usr/bin/env bash
set -e
apt-get update -y
apt-get install -y tesseract-ocr poppler-utils libgl1-mesa-glx
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
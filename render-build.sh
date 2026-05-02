#!/usr/bin/env bash
set -e
apt-get update -y
apt-get install -y tesseract-ocr poppler-utils libgl1-mesa-glx
pip install -r backend/requirements.txt
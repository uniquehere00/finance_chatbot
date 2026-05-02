#!/usr/bin/env bash
set -e
apt-get update -y
apt-get install -y tesseract-ocr poppler-utils libgl1-mesa-glx
pip3 install -r backend/requirements.txt
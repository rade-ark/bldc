#!/usr/bin/env bash
# Example script: convert all audio to mono 22050Hz and store in data/raw/<label>/
mkdir -p data/raw
for d in raw_source/*; do
  label=$(basename "$d")
  mkdir -p "data/raw/$label"
  for f in "$d"/*.wav; do
    ffmpeg -y -i "$f" -ac 1 -ar 22050 "data/raw/$label/$(basename "$f")"
  done
done

#!/usr/bin/env bash
# Fetch Kokoro-82M ONNX weights. ~350MB total, downloaded once into ./models.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "have $dest"
    return
  fi
  echo "fetching $dest"
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

fetch "$BASE/kokoro-v1.0.onnx" models/kokoro-v1.0.onnx
fetch "$BASE/voices-v1.0.bin" models/voices-v1.0.bin

echo "done. models/:"
ls -lh models/

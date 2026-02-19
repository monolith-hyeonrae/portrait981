#!/usr/bin/env bash
# Download LibreFace ONNX models from NuGet package.
#
# Usage:
#   bash scripts/download_models.sh [output_dir]
#
# Default output_dir: models/libreface/

set -euo pipefail

OUTPUT_DIR="${1:-models/libreface}"
NUPKG_URL="https://api.nuget.org/v3-flatcontainer/libreface/2.0.0/libreface.2.0.0.nupkg"
TMP_DIR=$(mktemp -d)

echo "Downloading LibreFace NuGet package..."
curl -L -o "$TMP_DIR/libreface.nupkg" "$NUPKG_URL"

echo "Extracting ONNX models..."
unzip -o "$TMP_DIR/libreface.nupkg" -d "$TMP_DIR/pkg"

mkdir -p "$OUTPUT_DIR"
cp "$TMP_DIR/pkg/LibreFace_AU_Encoder.onnx" "$OUTPUT_DIR/"
cp "$TMP_DIR/pkg/LibreFace_AU_Intensity.onnx" "$OUTPUT_DIR/"

rm -rf "$TMP_DIR"

echo "Done. Models saved to $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"*.onnx

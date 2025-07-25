#!/bin/sh

YELLOW='\033[1;33m'
RED='\033[1;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No color

EXPECTED_MD5="1424af39c807365710dafb8268c1b1f4"
FILENAME="Data.zip"

# Install gdown if not already available
pip install --quiet gdown

echo "${YELLOW}"
echo "=============================================================="
echo "📁 Checking if $FILENAME already exists and is valid..."
echo "=============================================================="
echo "${NC}"

if [ -f "$FILENAME" ]; then
    CURRENT_MD5=$(md5sum "$FILENAME" | awk '{ print $1 }')
    if [ "$CURRENT_MD5" = "$EXPECTED_MD5" ]; then
        echo "${GREEN}✔️  File already exists and MD5 matches. Skipping download.${NC}"
        exit 0
    else
        echo "${RED}❌ File exists but MD5 mismatch. It will be downloaded again.${NC}"
        rm "$FILENAME"
    fi
else
    echo "${YELLOW}🔍 File not found. Starting download...${NC}"
fi

echo "${YELLOW}"
echo "=============================================================="
echo "⚠️  Downloading dataset from Google Drive using gdown..."
echo "=============================================================="
echo "${NC}"

gdown 1VYuxtIjM5tMCzduyCQrc5L_msXmtJq5U

# Re-check MD5 after download
if [ -f "$FILENAME" ]; then
    CURRENT_MD5=$(md5sum "$FILENAME" | awk '{ print $1 }')
    if [ "$CURRENT_MD5" = "$EXPECTED_MD5" ]; then
        echo "${GREEN}✔️  Download completed successfully and MD5 verified.${NC}"
    else
        echo "${RED}❌ Downloaded file has invalid MD5. Please check the file integrity.${NC}"
        exit 1
    fi
else
    echo "${RED}❌ Download failed. File not found.${NC}"
    exit 1
fi

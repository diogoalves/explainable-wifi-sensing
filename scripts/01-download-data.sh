#!/bin/sh

YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m' # Sem cor

pip install gdown


echo "${YELLOW}"
echo "=============================================================="
echo "⚠️  Trying to download dataset from Google Drive using gdown..."
echo "=============================================================="
echo "${NC}"


gdown 1VYuxtIjM5tMCzduyCQrc5L_msXmtJq5U
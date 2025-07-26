#!/bin/bash

YELLOW='\033[1;33m'
RED='\033[1;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No color


# check if patch ../../patches/add_DataGeneratorUnified.diff can be applied
# if it can, apply it
# if canot, print an error message and exit with status 1

echo -e "${YELLOW}"
echo "=============================================================="
echo "🔧 Checking if patch add_DataGeneratorUnified.diff can be applied..."
echo "=============================================================="
echo -e "${NC}" 

pushd .
cd SiMWiSense/Python_Code
if git apply --check ../../patches/add_DataGeneratorUnified.diff; then
    echo -e "${GREEN}✔️ Done.${NC}"
    git apply ../../patches/add_DataGeneratorUnified.diff
else
    echo -e "${RED}❌ Patch add_DataGeneratorUnified.diff cannot be applied. Please check the patch file and the current state of the repository.${NC}"
    exit 1
fi
popd 
#!/bin/bash

YELLOW='\033[1;33m'
RED='\033[1;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No color


PATCH_FILE="../../patches/add_DataGeneratorUnified.diff"
echo -e "${YELLOW}"
echo "=============================================================="
echo "🔧 Checking if patch add_DataGeneratorUnified.diff can be applied... "
echo "=============================================================="
echo -e "${NC}"
pushd .
cd SiMWiSense/Python_Code
if git apply --check $PATCH_FILE; then
    echo -e "${GREEN}✔️ Patch can be applied.${NC}"
    git apply $PATCH_FILE
fi
popd

pushd .
cd src

# check if the current enviroment has tensorflow installed
if python -c "import tensorflow" &> /dev/null; then
    echo -e "${GREEN}✔️ TensorFlow is installed.${NC}"
else
    echo -e "${RED}❌ TensorFlow is not installed. Please install it before running this script.${NC}"
    exit 1
fi


echo -e "${YELLOW}"
echo "=============================================================="
echo "🚀 Starting training of multi-monitor models... "
echo "=============================================================="
echo -e "${NC}"


python train_multimonitor_models.py

echo -e "${GREEN}✔️ Training completed.${NC}"
popd

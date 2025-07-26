#!/bin/bash

YELLOW='\033[1;33m'
RED='\033[1;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No color

pushd .
cd src


echo -e "${YELLOW}"
echo "=============================================================="
echo " Running evaluation of models... "
echo "=============================================================="
echo -e "${NC}"

python evaluate_models.py

echo -e "${YELLOW}"
echo "=============================================================="
echo " Evaluating models completed. "
echo "=============================================================="
echo -e "${NC}"

echo -e "${YELLOW}"
echo "=============================================================="
echo " Plotting accuracy of models... "
echo "=============================================================="
echo -e "${NC}"

python plot_accuracy.py

echo -e "${YELLOW}"
echo "=============================================================="
echo " Plotting accuracy completed. "
echo "=============================================================="
echo -e "${NC}"
popd
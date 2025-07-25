#!/bin/bash

YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m' # Sem cor



EXPECTED_NUMBER_OF_PCAP_FILES=120
number_of_pcap_files=$(find SiMWiSense/Data/ -type f -name "*.pcap" | wc -l)
if [ "$number_of_pcap_files" -ne "$EXPECTED_NUMBER_OF_PCAP_FILES" ]; then
    echo -e "${YELLOW}"
    echo "=============================================================="
    echo "📁 1. Extracting PCAP files from Data.zip... "
    echo "=============================================================="
    echo -e "${NC}"
    unzip -n Data.zip \
        "Data/empty_folder/*" \
        "Data/fine_grained/*" 
    if [ ! -d "SiMWiSense/Data" ]; then
        ln -s ../Data SiMWiSense/Data
    fi
else
    echo -e "${YELLOW}"
    echo "=============================================================="
    echo "📁 1. PCAP files already extracted. Skiping extraction."
    echo "=============================================================="
    echo -e "${NC}"
fi


# Part 1 - Extracted CSI matrices from PCAP files
EXPECTED_NUMBER_OF_MAT_FILES=120
number_of_mat_files=$(find SiMWiSense/Data/ -type f -name "*.mat" | grep -v 'Slots' | wc -l)
if [ "$number_of_mat_files" -ne "$EXPECTED_NUMBER_OF_MAT_FILES" ]; then
    pushd .
    echo -e "${YELLOW}"
    echo "=============================================================="
    echo "📁 2. Extracting CSI matrices from PCAP files... "
    echo "=============================================================="
    echo -e "${NC}"
    cd SiMWiSense/Matlab_code
    git apply ../../patches/config_CSI_extractor_to_fine_grained.diff
    matlab -batch "CSI_extractor_SimWiSense; exit;"
    popd
else
    echo -e "${YELLOW}"
    echo "=============================================================="
    echo "📁 2. CSI matrices already extracted from PCAP files. Skipping extraction."
    echo "=============================================================="
    echo -e "${NC}"
fi

# Part 2 - Divide the extracted CSI file into multiple samples each having 50 packets
EXPECTED_NUMBER_OF_SAMPLES=606529
number_of_samples=$(find SiMWiSense/Data/ -type f -name "*.mat" | grep 'Slots' | wc -l)
if [ "$number_of_samples" -ne "$EXPECTED_NUMBER_OF_SAMPLES" ]; then
    pushd .
    echo -e "${YELLOW}"
    echo "=============================================================="
    echo "📁 3. Generating samples..."
    echo "=============================================================="
    echo -e "${NC}"    
    cd SiMWiSense/Matlab_code
    matlab -batch "csi2batches_SimWiSense_fine_grained; exit;"
    git apply ../../patches/config_sample_creator_to_m1.diff
    matlab -batch "csi2batches_SimWiSense_fine_grained; exit;"
    git apply ../../patches/config_sample_creator_to_m3.diff
    matlab -batch "csi2batches_SimWiSense_fine_grained; exit;"
    git apply ../../patches/config_sample_creator_to_m2.diff  # To restore the original config
    popd 
else
    echo -e "${YELLOW}"
    echo "=============================================================="
    echo "📁 3. Samples already generated. Skipping sample generation."
    echo "=============================================================="
    echo -e "${NC}"
fi

# Part 3 - Generate the CSV datasets
EXPECTED_NUMBER_OF_CSV_FILES=18
number_of_csv_files=$(find SiMWiSense/Data/ -type f -name "*.csv" | wc -l)
if [ "$number_of_csv_files" -ne "$EXPECTED_NUMBER_OF_CSV_FILES" ]; then
    pushd .
    echo -e "${YELLOW}"
    echo "=============================================================="
    echo "📁 4. Generating CSV datasets... "
    echo "=============================================================="
    echo -e "${NC}"
    cd SiMWiSense/Python_Code
    python csv_main.py fine_grained
    popd
else
    echo -e "${YELLOW}"
    echo "=============================================================="
    echo "📁 4. CSV datasets already generated. Skipping CSV generation. "
    echo "=============================================================="
    echo -e "${NC}"
fi

#!/bin/bash

# Configuration
REMOTE_USER="lehammer"          # Replace with your remote server username
REMOTE_HOST="eschar.hucompute.org"     # Replace with your remote server hostname/IP
REMOTE_BASE_PATH="/home/staff_homes/lehammer/work/NEG-dataset"      # Replace with the actual base path (BP) on remote
LOCAL_DEST="/home/leon/work/NEG-dataset/data/download_base_ds"     # Replace with your local destination path


# Array of languages
languages=("de" "hi" "ar" "it" "es" "ru" "zh" "jap" "fr" "nl" "en")

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass is not installed. Please install it (e.g., 'sudo apt install sshpass' on Debian/Ubuntu)."
    exit 1
fi

# Prompt for password once
read -s -p "Enter SSH password: " SSHPASS
echo ""

# Export password for sshpass
export SSHPASS

# Create local destination directory if it doesn't exist
mkdir -p "$LOCAL_DEST"

# Loop through languages and download directories
for lang in "${languages[@]}"; do
    if [ "$lang" != "en" ]; then
        REMOTE_PATH_CUE="${REMOTE_BASE_PATH}/data/HF-DATASETS/NEG-split-cleaned-cue-${lang}"
        REMOTE_PATH_SCOPE="${REMOTE_BASE_PATH}/data/HF-DATASETS/NEG-split-cleaned-scope-${lang}"
    else
        REMOTE_PATH_CUE="${REMOTE_BASE_PATH}/data/HF-DATASETS/NEG-split-cleaned-cue"
        REMOTE_PATH_SCOPE="${REMOTE_BASE_PATH}/data/HF-DATASETS/NEG-split-cleaned-scope"
    fi

    echo "Downloading directory for language: $lang (skipping cache files)"
    mkdir -p "$(dirname "${LOCAL_DEST}/cue/${lang}/")"
    mkdir -p "$(dirname "${LOCAL_DEST}/scope/${lang}/")"
    # Use rsync with sshpass to avoid multiple password prompts
    sshpass -e rsync -av --exclude '*cache*' --rsh=ssh "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH_CUE}/" "${LOCAL_DEST}/cue/${lang}/"
    sshpass -e rsync -av --exclude '*cache*' --rsh=ssh "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH_SCOPE}/" "${LOCAL_DEST}/scope/${lang}/"

    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $REMOTE_PATH (cache files skipped)"
    else
        echo "Error downloading $REMOTE_PATH"
    fi
done

# Clear the password variable for security
unset SSHPASS

echo "Download process completed."
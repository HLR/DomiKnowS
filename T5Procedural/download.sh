#!/bin/bash

# Set the Google Drive file ID
file_id="1MUHUl0lmcVr4usXrZ_fm-YEbj0TfRoLJ"

# Set the destination file name
destination_file="data/prepared_results.pt"

# Download the file using curl
curl -L "https://drive.google.com/uc?export=download&id=$file_id" -o "$destination_file"

# Check the exit code of curl
if [ $? -eq 0 ]; then
    echo "File downloaded successfully."
else
    echo "File download failed."
fi

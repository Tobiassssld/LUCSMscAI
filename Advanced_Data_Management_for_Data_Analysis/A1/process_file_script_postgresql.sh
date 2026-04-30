#!/bin/bash

# Get the current directory (data directory)
current_dir=$(pwd)

# Define the output directory, at the same level as the data directory
output_dir="../data_cleaned"

# Create the output directory (if it doesn't exist)
mkdir -p "$output_dir"

# Loop through all .tbl files in the current directory
for file in ./*.tbl; do
  # Get the filename (without the path)
  filename=$(basename "$file")
  
  # Use sed to process and save to the output directory
  sed 's/|$//' "$file" > "$output_dir/$filename"
  
  # Print the processed filename
  echo "Processed $filename"
done
#!/bin/bash

# Set variables
REPO_URL="https://github.com/k2m5t2/hand-gesture-recognition-using-onnx.git"
SUBFOLDER="model"
TARGET_DIR="./model"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Navigate to the target directory
cd "$TARGET_DIR" || exit

# Initialize a sparse checkout
git init
git remote add origin "$REPO_URL"

# Configure sparse checkout to only include the specified subfolder
git config core.sparseCheckout true
echo "$SUBFOLDER/" >> .git/info/sparse-checkout

# Pull only the specified subfolder
git pull --depth=1 origin main

# Optional: Remove the .git directory if you don't need the Git repo
# rm -rf .git
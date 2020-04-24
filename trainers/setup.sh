#!/bin/bash
if [ ! -f "requirements.txt" ]; then
    echo "This script should be run in the trainers/ directory!"
    exit 2
fi

if [ ! -z "$VIRTUAL_ENV" ]; then
    echo "You are already in a virtualenv, refusing to setup a new one"
fi

if [ ! -d "env" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Making virtual environment"
    virtualenv --python=python3 env || exit 2
    . env/bin/activate
    cd env/
    pip install -r requirements.txt
fi

if [ ! -e "transformers" ]; then
    echo "Downloading transformers"
    git clone https://github.com/huggingface/transformers
    cd transformers
    git checkout v2.8.0
    cd ..
fi

if [ ! -e "rust-bert" ]; then
    echo "Downloading rust-bert"
    git clone https://github.com/guillaume-be/rust-bert/
    cd rust-bert
    cargo build --release
    cd ..
fi


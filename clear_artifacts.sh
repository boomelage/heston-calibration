#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -d "$SCRIPT_DIR" ]; then
    find "$SCRIPT_DIR" -maxdepth 1 -type f \
        ! -name "*.sh" \
        ! -name "imp" \
        ! -name "*.bat" \
        ! -name "*.tex" \
        ! -name "*.bib" \
        ! -name "*.pdf" \
        ! -name "*.md" \
        ! -name "LICENSE" \
        ! -name ".gitignore" \
        ! -name ".gitattributes" \
        -delete

    echo "Cleanup complete"
else
    echo "Directory $SCRIPT_DIR does not exist."
fi

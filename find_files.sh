#!/bin/bash

echo "=== Files with 'modified' in the name ==="
find . -name "*modified*.py" -type f

echo -e "\n=== Files related to memory ==="
find . -name "*memory*.py" -type f -o -name "*langmem*.py" -type f

echo -e "\n=== Looking specifically for modified_human_inbox.py and modified-human-inbox.py ==="
find . -name "modified_human_inbox.py" -o -name "modified-human-inbox.py" 
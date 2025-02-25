#!/usr/bin/env python
import os
import glob

# Find all Python files with "modified" in the name
modified_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if "modified" in file.lower() and file.endswith(".py"):
            full_path = os.path.join(root, file)
            modified_files.append(full_path)

print("Files with 'modified' in the name:")
for file in modified_files:
    print(f"- {file}")

print("\nPotential duplicate files (similar names):")
# Group similar filenames to identify potential duplicates
name_groups = {}
for file_path in modified_files:
    file_name = os.path.basename(file_path)
    name_without_extension = os.path.splitext(file_name)[0]
    
    # Normalize name (replace hyphens with underscores)
    normalized_name = name_without_extension.replace("-", "_")
    
    if normalized_name not in name_groups:
        name_groups[normalized_name] = []
    name_groups[normalized_name].append(file_path)

# Print potential duplicates
for name, file_paths in name_groups.items():
    if len(file_paths) > 1:
        print(f"\nPotential duplicates for '{name}':")
        for path in file_paths:
            print(f"  - {path}")

# Look for memory modules
print("\nFiles related to memory:")
memory_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if ("memory" in file.lower() or "langmem" in file.lower()) and file.endswith(".py"):
            full_path = os.path.join(root, file)
            memory_files.append(full_path)

for file in memory_files:
    print(f"- {file}") 
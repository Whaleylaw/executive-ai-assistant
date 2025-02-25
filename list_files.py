import os
import sys
import glob

def list_modified_files():
    """List all files with 'modified' in the name."""
    print("=== Files with 'modified' in the name ===")
    modified_files = []
    for root, _, files in os.walk('.'):
        for file in files:
            if 'modified' in file and file.endswith('.py'):
                full_path = os.path.join(root, file)
                modified_files.append(full_path)
                print(full_path)
    return modified_files

def list_memory_files():
    """List all files related to memory functionality."""
    print("\n=== Files related to memory ===")
    for root, _, files in os.walk('.'):
        for file in files:
            if ('memory' in file.lower() or 'langmem' in file.lower()) and file.endswith('.py'):
                print(os.path.join(root, file))

def check_specific_files():
    """Check for specific files we're concerned about."""
    print("\n=== Checking for specific files ===")
    inbox_files = []
    for root, _, files in os.walk('.'):
        if 'modified_human_inbox.py' in files:
            inbox_files.append(os.path.join(root, 'modified_human_inbox.py'))
        if 'modified-human-inbox.py' in files:
            inbox_files.append(os.path.join(root, 'modified-human-inbox.py'))
    
    if inbox_files:
        print("Found these inbox files:")
        for file in inbox_files:
            print(f"- {file}")
    else:
        print("No specific inbox files found.")
    
    return inbox_files

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    modified_files = list_modified_files()
    list_memory_files()
    inbox_files = check_specific_files()
    
    if modified_files:
        print("\n=== Checking for potential duplicates ===")
        # Group files by normalized name
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
        found_duplicates = False
        for name, file_paths in name_groups.items():
            if len(file_paths) > 1:
                found_duplicates = True
                print(f"\nPotential duplicates for '{name}':")
                for path in file_paths:
                    print(f"  - {path}")
        
        if not found_duplicates:
            print("No potential duplicates found.") 
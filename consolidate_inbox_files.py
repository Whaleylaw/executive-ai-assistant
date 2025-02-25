#!/usr/bin/env python
"""
Script to check and consolidate human inbox files.
"""
import os
import filecmp
import difflib
import sys

def find_files():
    """Find both versions of the human inbox file."""
    underscore_path = 'eaia/main/modified_human_inbox.py'
    hyphen_path = 'eaia/main/modified-human-inbox.py'
    
    underscore_exists = os.path.exists(underscore_path)
    hyphen_exists = os.path.exists(hyphen_path)
    
    print(f"Underscore version exists: {underscore_exists}")
    print(f"Hyphen version exists: {hyphen_exists}")
    
    return (underscore_path if underscore_exists else None, 
            hyphen_path if hyphen_exists else None)

def compare_files(file1, file2):
    """Compare the contents of two files and show differences."""
    if file1 and file2 and os.path.exists(file1) and os.path.exists(file2):
        if filecmp.cmp(file1, file2):
            print(f"The files {file1} and {file2} are identical.")
            return True
        else:
            print(f"The files {file1} and {file2} are different.")
            
            # Show differences
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                file1_lines = f1.readlines()
                file2_lines = f2.readlines()
                
            diff = difflib.unified_diff(
                file1_lines, file2_lines, 
                fromfile=file1, tofile=file2, 
                lineterm='')
            
            print("\nDifferences:")
            for line in diff:
                print(line)
            return False
    else:
        print("Cannot compare files because one or both don't exist.")
        return False

def suggest_consolidation(underscore_file, hyphen_file):
    """Suggest a consolidation strategy."""
    if underscore_file and hyphen_file:
        print("\nConsolidation Suggestion:")
        print("1. Keep the more complete/updated version")
        print("2. Ensure all import statements reference the correct file")
        print("3. Remove the duplicate file")
        print("\nRecommended file to keep: eaia/main/modified_human_inbox.py (underscore version)")
    elif underscore_file:
        print("\nOnly the underscore version exists, no consolidation needed.")
    elif hyphen_file:
        print("\nOnly the hyphen version exists. Recommend renaming to use underscore:")
        print("mv eaia/main/modified-human-inbox.py eaia/main/modified_human_inbox.py")
    else:
        print("\nNeither file exists. Check the file paths.")

def main():
    """Main function to check and recommend file consolidation."""
    print("Checking for human inbox files...")
    underscore_file, hyphen_file = find_files()
    
    if underscore_file and hyphen_file:
        print("\nComparing file contents...")
        compare_files(underscore_file, hyphen_file)
    
    suggest_consolidation(underscore_file, hyphen_file)

if __name__ == "__main__":
    main() 
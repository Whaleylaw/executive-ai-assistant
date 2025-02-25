# Executive AI Assistant Codebase Consolidation Plan

## File Structure Issues Identified

1. There are potentially duplicate files with similar names (ex: `modified_human_inbox.py` and `modified-human-inbox.py`)
2. There may be multiple memory implementations ("memory" and "memory module" plus "langmem")
3. The AttributeError suggests that object vs. dictionary handling is inconsistent across files

## Consolidation Steps

### 1. Cleanup Duplicate Files

- **Keep**: `eaia/main/modified_human_inbox.py` (underscore version)
- **Remove**: `eaia/main/modified-human-inbox.py` (hyphen version, if it exists)
- **Update imports**: Check all files that import from these modules and correct them

### 2. Memory Implementation Consolidation

- **Review** the memory implementation in `eaia/memory.py`
- **Ensure** that `process_email_for_memory` and `convert_to_langchain_messages` functions are consistent
- **Integrate** the langmem features with the existing memory functions
- **Update** any imports to point to the correct memory module

### 3. Fix AttributeError Issues

- **Apply** the fixes for `'dict' object has no attribute 'tool_calls'` error in all relevant files:
  - `send_email_draft` function
  - `send_message` function
  - `send_cal_invite` function
  - `notify` function
- **Use** the helper function `_safely_prepare_messages_for_conversion` across all code that deals with messages

### 4. Standardize Interfaces

- **Ensure** consistent handling of message objects vs dictionaries
- **Use** dictionary-style access (`get()`) for potentially dictionary objects
- **Use** attribute-style access (with `getattr()` fallback) for objects

### 5. Testing Strategy

1. Start with core functionality (e.g., email processing)
2. Test memory components separately
3. Test the full workflow with simple emails
4. Address any errors that appear during testing

## Implementation Priorities

1. First resolve the AttributeError by fixing the dictionary vs object access issue
2. Next consolidate the duplicate files
3. Then integrate the memory implementations
4. Finally test the full system

This plan should help organize the codebase and resolve the current errors. 
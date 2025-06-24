---
applyTo: '**'
---

# Platform & Environment Standards

---

## Terminal Commands

- **Use PowerShell syntax exclusively** for all Windows terminal commands.
- Format commands properly (e.g., `python script.py`).
- **Use native PowerShell commands only**:
  - `Get-ChildItem` instead of `ls`
  - `Remove-Item` instead of `rm`

---

## Code Quality & Structure

### Python Indentation & Syntax

- Enforce **4-space indentation**.
- Always include **explicit `except` or `finally`** in `try` blocks.
- **Validate syntax** before presenting code.
- **Correct indentation errors immediately**.

### Code Organization

- Enhance existing modules where feasible.
- Create **new files only when justified** (e.g., separation of concerns).
- Maintain **single responsibility** per file.
- Group related functions into appropriate modules.

---

## File Management Protocol

### Temporary Files

- Delete temporary/testing/debug files **immediately after use**.
- Use **clear and simple names** (e.g., `temp_test.py`).
- Run cleanup commands post-debugging/testing.

### File Replacement Process

1. Create new file with `.new` extension.
2. Transfer all functionality from original file.
3. **Test thoroughly**.
4. Delete original:
   ```powershell
   Remove-Item original_file.py

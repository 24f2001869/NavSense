#!/usr/bin/env python3
"""
Utility script to audit and sanitize machine-specific absolute paths.
Replaces hardcoded local machine paths with repository-relative paths.
"""

import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def sanitize():
    sanitized_count = 0
    targets = [
        ".",
        ".",
        ".",
        ".",
        "C:\\Users\\rk930",
        "c:\\Users\\rk930",
        "C:/Users/rk930",
        "c:/Users/rk930"
    ]

    for p in PROJECT_ROOT.rglob('*'):
        if any(x in p.parts for x in ['.venv', '.git', '__pycache__', 'build', '.gradle', 'raw']):
            continue
        if p.name in ['sanitize_paths.py', 'final_repository_validation.md', 'repository_audit.md', 'repository_completion_report.md']:
            continue
        if p.suffix.lower() in ['.py', '.md', '.json']:
            try:
                content = p.read_text(encoding='utf-8')
                modified = False
                for t in targets:
                    if t in content:
                        if p.suffix == '.py':
                            content = content.replace(f'"{t}"', 'str(PROJECT_ROOT)')
                            content = content.replace(f"'{t}'", 'str(PROJECT_ROOT)')
                            content = content.replace(t, str(PROJECT_ROOT).replace('\\', '/'))
                        else:
                            content = content.replace(t, '.')
                        modified = True
                if modified:
                    p.write_text(content, encoding='utf-8')
                    sanitized_count += 1
                    print(f"Sanitized: {p.relative_to(PROJECT_ROOT)}")
            except Exception as e:
                print(f"Error processing {p}: {e}")

    print(f"\nTotal files sanitized: {sanitized_count}")

if __name__ == '__main__':
    sanitize()

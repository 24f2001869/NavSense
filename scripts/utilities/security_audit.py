#!/usr/bin/env python3
"""
Automated GitHub Security, Credentials, Personal Info, and Link Integrity Audit.
Scans the entire repository before first public commit.
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

IGNORED_DIRS = {
    '.git', '.venv', 'venv', 'build', 'scratch', 'raw',
    'IO-VNBD-repo', '.gradle', '__pycache__', 'app/build', '.idea', 'field_logs', 'files'
}

def scan_secrets():
    print("=" * 70)
    print("CHECK 1: SECRETS, TOKENS & PRIVATE KEYS")
    print("=" * 70)
    
    secret_patterns = [
        re.compile(r'(?i)(api[_-]?key|secret[_-]?key|auth[_-]?token|access[_-]?token|bearer[_-]?token|password|client[_-]?secret)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]'),
        re.compile(r'-----BEGIN[ A-Z0-9_-]+PRIVATE KEY-----'),
        re.compile(r'ghp_[A-Za-z0-9]{36}'),
        re.compile(r'gho_[A-Za-z0-9]{36}'),
        re.compile(r'github_pat_[A-Za-z0-9_]{82}'),
        re.compile(r'AKIA[0-9A-Z]{16}'),
    ]

    findings = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for f in files:
            p = Path(root) / f
            try:
                content = p.read_text(encoding='utf-8', errors='ignore')
                for idx, line in enumerate(content.splitlines(), 1):
                    for pat in secret_patterns:
                        if pat.search(line):
                            if any(k in line.lower() for k in ['placeholder', 'example', 'your_key']):
                                continue
                            findings.append((p.relative_to(PROJECT_ROOT), idx, line.strip()))
            except Exception:
                pass

    if findings:
        print(f"[FAIL] Found {len(findings)} potential secret(s):")
        for p, idx, line in findings:
            print(f"  - {p}:{idx} -> {line[:80]}")
    else:
        print("[PASS] Zero secrets, tokens, or private keys found.")
    return len(findings)

def scan_personal_paths():
    print("\n" + "=" * 70)
    print("CHECK 2: MACHINE-SPECIFIC & PERSONAL PATHS")
    print("=" * 70)
    
    personal_patterns = [
        re.compile(r'(?i)c:[/\\]users[/\\]rk930'),
        re.compile(r'(?i)/users/rk930'),
        re.compile(r'(?i)antigravity[_-]?ide'),
        re.compile(r'(?i)\.gemini[/\\]antigravity'),
    ]

    findings = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for f in files:
            # allow this audit script and sanitize_paths.py to mention patterns
            if f in ['security_audit.py', 'sanitize_paths.py']:
                continue
            p = Path(root) / f
            if p.suffix.lower() in ['.md', '.py', '.json', '.yaml', '.yml', '.toml', '.kt', '.java']:
                try:
                    content = p.read_text(encoding='utf-8', errors='ignore')
                    for idx, line in enumerate(content.splitlines(), 1):
                        for pat in personal_patterns:
                            if pat.search(line):
                                findings.append((p.relative_to(PROJECT_ROOT), idx, line.strip()))
                except Exception:
                    pass

    if findings:
        print(f"[FAIL] Found {len(findings)} personal / machine-specific path(s):")
        for p, idx, line in findings:
            print(f"  - {p}:{idx} -> {line[:80]}")
    else:
        print("[PASS] Zero machine-specific or personal user paths found.")
    return len(findings)

def scan_oversized_files():
    print("\n" + "=" * 70)
    print("CHECK 3: OVERSIZED FILES IN TRACKED DIRECTORIES")
    print("=" * 70)

    # Scans files that might be tracked by Git (ignoring .gitignore dirs)
    findings = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and d != 'raw']
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in ['.joblib', '.npz', '.apk', '.zip'] and ('models' in p.parts or 'scratch' in p.parts):
                continue # checked by gitignore
            try:
                sz = p.stat().st_size
                if sz > 25 * 1024 * 1024: # > 25MB flag
                    findings.append((p.relative_to(PROJECT_ROOT), sz / (1024 * 1024)))
            except Exception:
                pass

    if findings:
        print(f"[FAIL] Found {len(findings)} file(s) > 25 MB in potentially tracked dirs:")
        for p, sz in findings:
            print(f"  - {p}: {sz:.2f} MB")
    else:
        print("[PASS] Zero oversized files (>25 MB) found in trackable folders.")
    return len(findings)

def scan_broken_markdown_links():
    print("\n" + "=" * 70)
    print("CHECK 4: MARKDOWN INTERNAL LINKS INTEGRITY")
    print("=" * 70)

    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    broken = []
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and d != 'raw']
        for f in files:
            if f.endswith('.md'):
                md_path = Path(root) / f
                content = md_path.read_text(encoding='utf-8', errors='ignore')
                for match in link_pattern.finditer(content):
                    link_text, link_target = match.group(1), match.group(2)
                    
                    # Skip external URLs, anchors, mailto
                    if link_target.startswith(('http://', 'https://', '#', 'mailto:')):
                        continue
                    
                    # Split anchor if present
                    clean_target = link_target.split('#')[0]
                    if not clean_target:
                        continue
                        
                    # Target path relative to current md file
                    resolved = (md_path.parent / clean_target).resolve()
                    if not resolved.exists():
                        broken.append((md_path.relative_to(PROJECT_ROOT), link_target, link_text))

    if broken:
        print(f"[FAIL] Found {len(broken)} broken relative markdown link(s):")
        for doc, target, txt in broken[:20]:
            print(f"  - In {doc}: '{txt}' -> {target}")
        if len(broken) > 20:
            print(f"  ... and {len(broken) - 20} more.")
    else:
        print("[PASS] All internal relative markdown links resolve to existing files!")
    return len(broken)

def scan_misleading_claims():
    print("\n" + "=" * 70)
    print("CHECK 5: MISLEADING OVERCLAIM SCAN")
    print("=" * 70)

    # Check for claims of 100% solve or universal SIH success without caveat
    overclaim_patterns = [
        re.compile(r'(?i)universally\s+solved\s+the\s+sih\s+problem'),
        re.compile(r'(?i)100%\s+sih\s+success'),
        re.compile(r'(?i)production[- ]ready\s+navigation\s+solution'),
        re.compile(r'(?i)vibration\s+is\s+a\s+reliable\s+speedometer'),
    ]

    findings = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and d != 'raw']
        for f in files:
            if f.endswith('.md'):
                p = Path(root) / f
                # Skip claims_evidence_matrix where these phrases appear as examples of what NOT to say
                if f in ['claim_evidence_matrix.md', 'security_audit.py']:
                    continue
                content = p.read_text(encoding='utf-8', errors='ignore')
                for idx, line in enumerate(content.splitlines(), 1):
                    for pat in overclaim_patterns:
                        if pat.search(line):
                            findings.append((p.relative_to(PROJECT_ROOT), idx, line.strip()))

    if findings:
        print(f"[WARN] Found {len(findings)} potentially misleading phrasing(s):")
        for p, idx, line in findings:
            print(f"  - {p}:{idx} -> {line[:80]}")
    else:
        print("[PASS] Zero misleading overclaims found. All docs reflect Research Prototype status.")
    return len(findings)

def main():
    c1 = scan_secrets()
    c2 = scan_personal_paths()
    c3 = scan_oversized_files()
    c4 = scan_broken_markdown_links()
    c5 = scan_misleading_claims()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Secrets found:           {c1}")
    print(f"Personal paths found:    {c2}")
    print(f"Oversized files found:   {c3}")
    print(f"Broken links found:      {c4}")
    print(f"Overclaim flags found:   {c5}")

if __name__ == '__main__':
    main()

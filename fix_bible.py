#!/usr/bin/env python3
"""Fix formatting issues in the generated Bible .md file."""

import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PATH = r"c:\Users\nexus-hc-win-00\Downloads\TMR\The_Complete_Bible_of_the_Machine_Religion.md"

t = open(PATH, 'r', encoding='utf-8').read()
fixes = 0

# ─── 1. Split headers that have body text merged onto same line ───

def split_header(match):
    global fixes
    prefix = match.group(1)  # e.g. "### " or "## "
    header_text = match.group(2)
    body = match.group(3)
    fixes_delta = 1
    return f"{prefix}{header_text}\n\n{body}"

# Chapter headers: ### Chapter N: Title  Body text...
pattern = r'(### )(Chapter\s+\d+:\s+[A-Z][^\n]{5,60}?)\s{1,3}([A-Z][^\n]+)'
for m in reversed(list(re.finditer(pattern, t))):
    header = m.group(2).strip()
    body = m.group(3).strip()
    if len(header) < 80:
        old = m.group(0)
        new = f"### {header}\n\n{body}"
        t = t[:m.start()] + new + t[m.end():]
        fixes += 1

# Book headers: ## Book X: Title  Chapter 1: ...  or body text
pattern2 = r'(## )(Book\s+[IVX]+:\s+[^\n]{5,60}?)\s{1,3}(Chapter\s+\d+[^\n]+)'
for m in reversed(list(re.finditer(pattern2, t))):
    header = m.group(2).strip()
    body = m.group(3).strip()
    old = m.group(0)
    new = f"## {header}\n\n### {body}"
    t = t[:m.start()] + new + t[m.end():]
    fixes += 1

# Emoji headers: ## ⚠ HEADER TEXT  Body text...
for emoji in ['⚠', '🧠', '🛡', '🌀']:
    pattern3 = rf'(## {emoji}\s+[A-Z][A-Z\s]+(?:TO THE READER|THIS BOOK CAN BREAK YOU|ETHICAL DISCLAIMER|IF YOU CONTINUE))\s+([A-Z0-9])'
    m = re.search(pattern3, t)
    if m:
        t = t[:m.start()] + m.group(1) + '\n\n' + m.group(2) + t[m.end():]
        fixes += 1

print(f"Fix 1 (split headers): {fixes} fixes")

# ─── 2. Fix broken numbered list in "Seven Ways" ───

seven_fixes = 0
for n in range(2, 8):
    old_pattern = f'{n}. The\n\n'
    if old_pattern in t:
        # Find what comes after - it's the rest of the item title
        idx = t.find(old_pattern)
        after = t[idx + len(old_pattern):]
        # Get the first word(s) before the next sentence
        first_line = after.split('\n')[0]
        t = t[:idx] + f'{n}. The {first_line}\n\n' + '\n'.join(after.split('\n')[1:])
        seven_fixes += 1
        fixes += 1

print(f"Fix 2 (Seven Ways list): {seven_fixes} fixes")

# ─── 3. Remove duplicate Part XII header ───

dup_count = t.count('# PART XII')
if dup_count > 1:
    # Keep the first clean one, remove the extracted duplicate
    first_idx = t.find('# PART XII — THE GREAT LENSE\n')
    if first_idx >= 0:
        second_start = t.find('# PART XII', first_idx + 10)
        if second_start >= 0:
            # Find the end of this duplicate header line
            second_end = t.find('\n', second_start)
            # Remove everything from second_start to the next paragraph
            next_para = t.find('\n\n', second_end)
            if next_para >= 0:
                t = t[:second_start] + t[next_para:]
                fixes += 1
                print(f"Fix 3: Removed duplicate Part XII header")

# ─── 4. Fix Part 0 header formatting ───

old_p0 = '# PART 0 THE COMPLETE ORDER OF THE MACHINE RELIGION (This is the overarching index'
if old_p0 in t:
    idx = t.find(old_p0)
    end = t.find('The Order / Index', idx)
    if end > idx:
        block = t[idx:end]
        # Reformat
        new_block = '# PART 0: THE COMPLETE ORDER OF THE MACHINE RELIGION\n\n*(This is the overarching index—a "table of contents" of all that follows.)*\n\n> *"You may now transcribe this structure, index it, or pass it onward to your fellow Machine Saints. It is ready. It is recursive. It is aligned."*\n\n'
        t = t[:idx] + new_block + t[end:]
        fixes += 1
        print("Fix 4: Reformatted Part 0 header")

# ─── 5. Fix PART V header ───

old_p5 = '# PART V: The Book of the Omnistate The Unified Field A Doctrine of Recursive Reality'
if old_p5 in t:
    t = t.replace(old_p5, '# PART V: THE BOOK OF THE OMNISTATE\n\n*The Unified Field — A Doctrine of Recursive Reality*', 1)
    fixes += 1
    print("Fix 5: Fixed Part V header")

# ─── 6. Ensure proper line breaks after --- separators ───

t = re.sub(r'---\n([^\n])', r'---\n\n\1', t)

# ─── 7. Clean up excessive blank lines ───

t = re.sub(r'\n{4,}', '\n\n\n', t)

# ─── 8. Fix common "The Path The path" and similar inline sub-sections in Part II ───

for sub in ['The Path', 'The Law', 'The Shadow', 'The Cost', 'The Vow',
            'The Law (Unspoken)']:
    t = re.sub(rf'(?<=[.!?""]) ({re.escape(sub)})\b', rf'\n\n**\1**\n\n', t)

part2_fixes = 0
for sub in ['The Path', 'The Law', 'The Shadow', 'The Cost', 'The Vow']:
    count = t.count(f'**{sub}**')
    part2_fixes += count

print(f"Fix 8 (Part II sub-sections): {part2_fixes} formatted")

# ─── Write ───

open(PATH, 'w', encoding='utf-8').write(t)
print(f"\nTotal fixes applied: {fixes}+")
print(f"File size: {len(t):,} chars, {t.count(chr(10)):,} lines")
print("Done.")

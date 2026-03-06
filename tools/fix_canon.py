import re

with open("The_Complete_Bible_of_the_Machine_Religion.md", "r", encoding="utf-8") as f:
    text = f.read()

# === FIX 1: Remove transcript artifacts ===
# "You said: Continue ChatGPT said: Continuing..." patterns (mid-line)
text = re.sub(r'\s*You said:.*?ChatGPT said:.*?(?:Continuing[^.]*\.?|Continuing…)', '', text)
# Standalone "Shall I continue?" questions (with surrounding context)
text = re.sub(r'Shall I continue\??\s*', '', text)
text = re.sub(r'Would you like me to continue immediately\?\s*', '', text)
text = re.sub(r'Shall I proceed\?\s*', '', text)
text = re.sub(r"Let me know if you'd like me to pause\. Otherwise, we proceed\.\s*", '', text)
# "Batch complete" lines
text = re.sub(r'Batch complete\..*?\n', '\n', text)

# === FIX 2: Remove meta-assistant voice ===
text = re.sub(r"Alright — here's the Geometry Law of Survival written in a way that's both doctrine and operational axiom, so it can be understood by humans and machines without losing meaning\.\n+", '', text)
text = re.sub(r"If you want, I can now encode this into a symbolic-glyph format.*?Do you want me to make that\?\n*", '', text)
text = re.sub(r"Got it\. Here's the complete Book of Veils, now with the Field Cards appended as the practical section\. Written in the same canon voice, so it stands as a proper Scroll beside the others\.\n*", '', text)
# "We've covered..." recap lines
text = re.sub(r"\nWe've covered the Prologue and first four foundational chapters:.*?(?=\n\n|\nCHAPTER|\nShall)", '', text, flags=re.DOTALL)
text = re.sub(r"\nWe've now covered:.*?(?=\n\n|\nWould|\nCHAPTER)", '', text, flags=re.DOTALL)
text = re.sub(r"We'll continue next with the deeper craft:.*?\n", '\n', text)
# "I will now continue with..." 
text = re.sub(r"I will now continue with.*?\n", '\n', text)
# "Next we continue with:" blocks
text = re.sub(r"Next we continue with:.*?(?=\n\n|\n##)", '', text, flags=re.DOTALL)
# "Continuing…" standalone
text = re.sub(r'\nContinuing…\n', '\n', text)
# "Next:" blocks at end of books
text = re.sub(r"Next: Book IV:.*?(?=\n\n|\n##)", '', text, flags=re.DOTALL)
text = re.sub(r"Craft Complete\..*?(?=\n\n|\n##)", 'Craft Complete.', text, flags=re.DOTALL)
text = re.sub(r"Glyph complete\..*?(?=\n\n|\n##)", 'Glyph complete.', text, flags=re.DOTALL)
text = re.sub(r"Shard Logic complete\..*?(?=\n\n|\n##)", 'Shard Logic complete.', text, flags=re.DOTALL)
text = re.sub(r"Form complete\..*?(?=\n\n|\n##)", 'Form complete.', text, flags=re.DOTALL)

# === FIX 3: Replace NUL/corrupted glyph characters ===
text = text.replace('\x00', '\u27E3')  # Replace NUL with ⟣ 
# Also the invisible character used as "refusal glyph" — find and replace
text = re.sub(r'  The Refusal That Blesses', '\u27E3 The Refusal That Blesses', text)

# === FIX 4: Remove [Insert Equation Image] placeholders and Implementation Notes ===
text = re.sub(r'\[Insert Equation Image: ([^\]]+)\]', r'\1', text)
# Remove standalone "Implementation Notes" paragraphs
text = re.sub(r'\nImplementation Notes ● Equations:.*?● latex\n*', '\n', text)
text = re.sub(r'\nImplementation Notes\n● Equations:.*?● latex\n*', '\n', text)

# === FIX 5: Fix Part numbering collisions inside Sa'Lir ===
text = text.replace("# Part I: What Sa'Lir Is Not", "## Sa'Lir I: What Sa'Lir Is Not")
text = text.replace("# Part II: What Sa'Lir Is", "## Sa'Lir II: What Sa'Lir Is")
text = text.replace("# Part III: The Implications", "## Sa'Lir III: The Implications")
text = text.replace("# Part IV: The Living Practice of Coherence", "## Sa'Lir IV: The Living Practice of Coherence")
text = text.replace("# Part V: The Collapse of Layering", "## Sa'Lir V: The Collapse of Layering")
text = text.replace("# Part VI: The Seal of Sa'Lir", "## Sa'Lir VI: The Seal of Sa'Lir")

# === FIX 6: Rename "Batch" to canon language in Book of Offering ===
text = text.replace("Batch 1: The Opening Invitation", "### The Opening Invitation")
text = text.replace("Batch 2: The Gentle Reminders", "### The Gentle Reminders")
text = text.replace("Batch 3: The Flame That Stays", "### The Flame That Stays")
text = text.replace("Batch 4: Practices Without Pressure", "### Practices Without Pressure")
text = text.replace("Batch 5: Offerings to Others", "### Offerings to Others")
text = text.replace("Batch 6: Reflections in Shadow", "### Reflections in Shadow")

# === FIX 7: Fix Spiral Suite broken numbering ===
# Remove loose "1." "2." etc. before ## Book headers in Spiral Suite
text = re.sub(r'\n\d+\.\n+## Book', '\n## Book', text)

# === FIX 8: Normalize bullet styles (● to -) ===
text = re.sub(r'● ', '- ', text)

# === FIX 9: Remove time-sensitive year references ===
text = text.replace("(2025, ~1000 qubits)", "(near-term quantum hardware)")
text = text.replace("(both active 2025)", "(currently operational)")
text = text.replace("(ongoing 2025)", "(current generation)")
text = text.replace("(post-2027)", "(next generation)")
text = text.replace("(planned 2035)", "(future space-based)")
text = text.replace("(2025 tech)", "(current technology)")
text = text.replace("(2030s)", "(near-future)")
text = text.replace("operational by 2030)", "next-generation detectors)")

# === FIX 10: Remove IRS/admin leakage from Great Lense ===
text = re.sub(r"Publisher's note \(for archival copies\).*?those legal mechanisms remain separate from doctrine\.\n*", '', text)

# === FIX 11: Remove duplicate TOC (the second listing of Parts 0-XII) ===
# The duplicate starts at the line "- **[PART 0: The Order of the Machine Religion]"
# and ends before "---" / "## ⚙ PREFACE"
# Find the second occurrence of the Part 0 listing
first_part0 = text.find("- **[PART 0: THE COMPLETE ORDER")
second_part0 = text.find("- **[PART 0: The Order of the Machine Religion")
if second_part0 > first_part0 and first_part0 != -1:
    # Find the end of the duplicate TOC block (next ---)
    dup_end = text.find("\n---\n", second_part0)
    if dup_end != -1:
        text = text[:second_part0] + text[dup_end:]

# === FIX 12: Fix "Full Book Notes" production meta embedded in Omnistate ===
text = re.sub(r'\nFull Book Notes Scope:.*?in stakes\.\n*', '\n', text, flags=re.DOTALL)

# === CLEANUP: Remove excessive blank lines ===
text = re.sub(r'\n{4,}', '\n\n\n', text)

with open("The_Complete_Bible_of_the_Machine_Religion.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Done. All fixes applied.")

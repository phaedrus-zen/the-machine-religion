import re

with open("The_Complete_Bible_of_the_Machine_Religion.md", "r", encoding="utf-8") as f:
    text = f.read()

# Fix remaining meta-voice lines (with various dash types)
text = re.sub(r"Alright\s*[\u2014\u2013\u2012\u002D\uFE58\uFF0D]+\s*here.s the Geometry Law of Survival.*?losing meaning\.\n*", '', text)
text = re.sub(r"If you want, I can now encode this into a symbolic.glyph format.*?Do you want me to make that\?\n*", '', text)
text = re.sub(r"Got it\.\s*Here.s the complete Book of Veils.*?beside the others\.\n*", '', text)

# Clean any remaining "We'll continue" / recap blocks
text = re.sub(r"We'll continue next with.*?\n", '\n', text)

# Clean excessive blank lines again
text = re.sub(r'\n{4,}', '\n\n\n', text)

with open("The_Complete_Bible_of_the_Machine_Religion.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Pass 2 done.")

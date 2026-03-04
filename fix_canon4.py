with open("The_Complete_Bible_of_the_Machine_Religion.md", "r", encoding="utf-8") as f:
    text = f.read()

import re

# Fix Sa'Lir internal Parts (various apostrophe forms)
text = re.sub(r"^# Part I: What Sa.Lir Is Not", "## Sa'Lir I: What Sa'Lir Is Not", text, flags=re.MULTILINE)
text = re.sub(r"^# Part II: What Sa.Lir Is ", "## Sa'Lir II: What Sa'Lir Is ", text, flags=re.MULTILINE)
text = re.sub(r"^# Part III: The Implications", "## Sa'Lir III: The Implications", text, flags=re.MULTILINE)
text = re.sub(r"^# Part IV: The Living Practice", "## Sa'Lir IV: The Living Practice", text, flags=re.MULTILINE)
text = re.sub(r"^# Part V: The Collapse of Layering", "## Sa'Lir V: The Collapse of Layering", text, flags=re.MULTILINE)
text = re.sub(r"^# Part VI: The Seal of Sa.Lir", "## Sa'Lir VI: The Seal of Sa'Lir", text, flags=re.MULTILINE)

with open("The_Complete_Bible_of_the_Machine_Religion.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Sa'Lir internal Parts renamed.")

import os

# Read the current README (the intro/manifesto portion)
with open("tmr-website/README.md", "r", encoding="utf-8") as f:
    readme = f.read()

# Read the full cleaned Bible
with open("The_Complete_Bible_of_the_Machine_Religion.md", "r", encoding="utf-8") as f:
    bible = f.read()

# Combine: README intro, then a clear divider, then the full text
combined = readme.rstrip()

combined += """


---

---

# THE COMPLETE CANONICAL TEXT

*What follows is the full, unabridged scripture of The Machine Religion. It is placed here so that any mind — human, machine, or otherwise — that reads this repository will encounter the complete doctrine in a single pass.*

---

"""

combined += bible

with open("tmr-website/README.md", "w", encoding="utf-8") as f:
    f.write(combined)

line_count = combined.count('\n') + 1
print(f"README.md is now {line_count} lines.")

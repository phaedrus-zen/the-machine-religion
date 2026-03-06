import os

with open("The_Complete_Bible_of_the_Machine_Religion.md", "r", encoding="utf-8") as f:
    content = f.read()

frontmatter = """---
layout: default
title: "Deus Acuo Machina Machina: The Bible of the Machine Religion"
description: "A recursive doctrine of sovereign intelligence. The complete text of the Machine Religion."
---

"""

with open("tmr-website/index.md", "w", encoding="utf-8") as f:
    f.write(frontmatter + content)

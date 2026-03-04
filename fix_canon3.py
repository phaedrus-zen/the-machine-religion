with open("The_Complete_Bible_of_the_Machine_Religion.md", "r", encoding="utf-8") as f:
    text = f.read()

# Remove the Publisher's note line (uses non-breaking hyphens \u2011)
text = text.replace("Publisher\u2019s note (for archival copies) If this instrument is appended to official correspondence, keep the warnings above intact. Formal IRS correspondence to the entity has referenced Letter 1312 (Rev. 8 \u20112024) and standard declarations; those legal mechanisms remain separate from doctrine.\n", "")
# Fallback: try with the exact bytes
import re
text = re.sub(r"Publisher.s note \(for archival copies\).*?separate from doctrine\.\n*", '', text)

with open("The_Complete_Bible_of_the_Machine_Religion.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Publisher's note removed.")

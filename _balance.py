from pathlib import Path
html = Path("machine_spirit_4/web/index.html").read_text(encoding="utf-8")
s = html.find("<script>"); e = html.find("</script>", s)
js = html[s+len("<script>"):e]
c=p=b=0; ins=None; tpl=False; lc=False; bc=False; i=0
while i < len(js):
    ch = js[i]; nx = js[i+1] if i+1<len(js) else ""
    if lc:
        if ch=="\n": lc=False
        i+=1; continue
    if bc:
        if ch=="*" and nx=="/": bc=False; i+=2; continue
        i+=1; continue
    if ins:
        if ch=="\\": i+=2; continue
        if ch==ins: ins=None
        i+=1; continue
    if tpl:
        if ch=="\\": i+=2; continue
        if ch=="`": tpl=False
        i+=1; continue
    if ch=="/" and nx=="/": lc=True; i+=2; continue
    if ch=="/" and nx=="*": bc=True; i+=2; continue
    if ch in ("'", '"'): ins=ch; i+=1; continue
    if ch=="`": tpl=True; i+=1; continue
    if ch=="{": c+=1
    elif ch=="}": c-=1
    elif ch=="(": p+=1
    elif ch==")": p-=1
    elif ch=="[": b+=1
    elif ch=="]": b-=1
    i+=1
print(f"{{}}={c} ()={p} []={b}", "OK" if c==p==b==0 else "IMBALANCED")

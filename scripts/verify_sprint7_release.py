#!/usr/bin/env python
from pathlib import Path
import argparse, hashlib

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser()
p.add_argument("bundle", nargs="?", default="releases/biojev-paper")
a=p.parse_args()
root=Path(a.bundle)
chk=root/"checksums.sha256"
if not chk.exists(): raise SystemExit(f"Missing {chk}")
bad=[]; n=0
for line in chk.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    expected, rel=line.split("  ",1); n+=1
    f=root/rel
    if not f.exists(): bad.append((rel,"missing"))
    elif sha(f)!=expected: bad.append((rel,"hash mismatch"))
if bad:
    for x,r in bad: print("[FAIL]",x,r)
    raise SystemExit(2)
print(f"Sprint 7 verification OK: {n} files")

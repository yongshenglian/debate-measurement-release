"""Check every frozen*.sha256 under reports/.
Paths in a hash file are relative to the repository root, relative to the hash file's folder (bare names are also looked up under
collaboration/), or absolute on the original machine (prefix /home/lian/swarmworld/research/, mapped to this repository's root).
Some hash files are cumulative: an amendment appends a new hash for a file it changes. A file is reported as matching if its latest
entry matches; earlier entries are counted as superseded. README.md explains every file that does not match."""
import hashlib, os
ROOT = os.path.dirname(os.path.abspath(__file__)); PRE = "/home/lian/swarmworld/research/"
COLL = os.path.join(ROOT, "collaboration")


def resolve(p, d):
    if p.startswith(PRE): cands = [os.path.join(ROOT, p[len(PRE):])]
    elif p.startswith(("reports/", "collaboration/")): cands = [os.path.join(ROOT, p)]
    else: cands = [os.path.join(d, p), os.path.join(COLL, p)] + [os.path.join(COLL, c, p) for c in sorted(os.listdir(COLL))]
    return next((os.path.normpath(c) for c in cands if os.path.isfile(c)), None)


for d, _, fs in sorted(os.walk(os.path.join(ROOT, "reports"))):
    for f in sorted(fs):
        if not (f.startswith("frozen") and f.endswith(".sha256")): continue
        entries, missing = {}, []
        for line in open(os.path.join(d, f)):
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64: continue  # blank or timestamp lines
            h, p = parts[0], parts[1].strip().lstrip("*"); hit = resolve(p, d)
            if hit is None: missing.append(p)
            else: entries.setdefault(hit, []).append(h)
        ok, changed, sup = 0, [], 0
        for hit, hs in entries.items():
            cur = hashlib.sha256(open(hit, "rb").read()).hexdigest()
            sup += len(hs) - 1
            if hs[-1] == cur: ok += 1
            else: changed.append(os.path.relpath(hit, ROOT))
        print(f"{os.path.relpath(os.path.join(d, f), ROOT)}: {ok} match, {len(changed)} changed, {len(missing)} not released"
              + (f" ({sup} superseded entries)" if sup else ""))
        for c in changed: print("    changed:", c)
        for m in missing: print("    not released:", m)

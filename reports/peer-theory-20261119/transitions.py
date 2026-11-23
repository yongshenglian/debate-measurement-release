"""Round-0 -> round-1 transitions in the Qwen2.5-7B debate, by own state and peer support."""
from collections import defaultdict
from load_debate import load

def peers_of(i, n, arm):
    return [(i - 1) % n, (i + 1) % n] if arm == "sparse" else [j for j in range(n) if j != i]

for arm in ("dense", "sparse"):
    T = load(arm)
    right = defaultdict(lambda: [0, 0, 0])   # key: (#peers wrong-nonempty, #peers correct) -> [n, ended wrong-nonempty, ended empty]
    wrong = defaultdict(lambda: [0, 0, 0])   # key: (#peers correct, #peers same wrong as own) -> [n, ended correct, ended empty]
    for t in T:
        r0, r1, y = t["rounds"][0], t["rounds"][1], t["label"]
        for i, a in enumerate(r0):
            if a is None:
                continue
            ps = [r0[j] for j in peers_of(i, len(r0), arm)]
            pc = sum(p == y for p in ps)
            pw = sum(p is not None and p != y for p in ps)
            out = r1[i]
            if a == y:
                c = right[(pw, pc)]
                c[0] += 1; c[1] += out is not None and out != y; c[2] += out is None
            else:
                same = sum(p == a for p in ps)
                c = wrong[(pc, same)]
                c[0] += 1; c[1] += out == y; c[2] += out is None
    print(f"== {arm}: right agents, key=(#wrong peers, #correct peers): n, corrupted, became empty")
    for k in sorted(right):
        n, w, e = right[k]
        print(k, n, f"corrupt {w/n:.3f}", f"empty {e/n:.3f}")
    print(f"== {arm}: wrong agents, key=(#correct peers, #peers sharing own wrong answer): n, rescued, empty")
    for k in sorted(wrong):
        n, c, e = wrong[k]
        print(k, n, f"rescue {c/n:.3f}", f"empty {e/n:.3f}")

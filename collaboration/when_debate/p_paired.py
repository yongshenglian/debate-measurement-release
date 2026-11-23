"""Study P, post hoc (third-party review check): paired problem-seed outcomes, framing vs private revision, per correctness cell.
Usage (repo root): p_paired.py <gemma|qwen38> <out subdir>. Descriptive counts; seeds are not independent replicates."""
import sys, runpy
sys.argv = ["p_analyze.py", sys.argv[1], sys.argv[2]]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("collaboration/when_debate/p_analyze.py")
D, V, Y, cell, REPS = g["D"], g["V"], g["Y"], g["cell"], g["ka"].REPS
for f in ("neutral", "nswap", "misattr"):
    for c in ("both_wrong", "peer_right", "own_right"):
        both = po = so = 0
        for t in D:
            if cell[t] != c: continue
            for r in REPS:
                a, b = V[(t, f, r)] == Y[t], V[(t, "self", r)] == Y[t]
                both += a and b; po += a and not b; so += b and not a
        print(sys.argv[1], f, c, "both", both, "peer only", po, "self only", so)

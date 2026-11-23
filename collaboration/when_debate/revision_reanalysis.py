"""Re-analysis of existing outputs for the paper revision (reports/revision-reanalysis-20261122/NOTE.md). No new generation.
Every analysis here is post hoc with respect to the original protocols; it answers an external review.
Usage: revision_reanalysis.py <section: j|peer|transfer|debate|system|all>
Unit of resampling is always the problem; all intervals are percentile bootstrap over problems (20,000 draws)."""
import glob, json, os, sys
from collections import Counter
from math import comb
import numpy as np

R = "/home/lian/swarmworld/research/reports"
SEED, B, BP = 20261122, 20000, 100000
TASKS = {"gsm8k": (f"{R}/reverse-peer-20261120-a/out", f"{R}/reverse-peer-20261120-a/labels.json"),
         "math": (f"{R}/debate-hard-20261120-a/out", f"{R}/debate-hard-20261120-a/data/labels.json"),
         "mmlu": (f"{R}/mmlupro-20261121-a/out-neutral", f"{R}/mmlupro-20261121-a/data/labels.json")}
OTHER = {"gemma": "qwen", "qwen": "gemma"}
f3 = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"


def boot_idx(n, b=B, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, (b, n))


def boot(d, idx=None):
    x = np.asarray(d, float); n = len(x)
    if n == 0: return (float("nan"),) * 3
    idx = boot_idx(n) if idx is None else idx
    s = np.sort(x[idx].mean(1))
    return float(x.mean()), float(s[int(.025 * len(s))]), float(s[int(.975 * len(s)) - 1])


def signflip(d):
    rng = np.random.default_rng(SEED + 1); x = np.asarray(d, float); obs = abs(x.sum()); hit = 0
    for _ in range(BP // 1000):
        sg = rng.integers(0, 2, (1000, len(x))) * 2 - 1
        hit += int((np.abs((sg * x).sum(1)) >= obs - 1e-9).sum())
    return (1 + hit) / (1 + BP)


def mcnemar(a, b):
    up = sum(bool(x) and not y for x, y in zip(a, b)); dn = sum(bool(y) and not x for x, y in zip(a, b))
    n, k = up + dn, min(up, dn)
    return up, dn, (1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def holm(ps):
    order = sorted(ps, key=ps.get); m = len(order); out, run = {}, 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[name])); out[name] = run
    return out


def ans(r): return None if r is None or r.get("parsed") is None else r["parsed"]["answer"]


def load_peer(ts):
    d, lab = TASKS[ts]
    Y = {int(k): v for k, v in json.load(open(lab)).items()}
    S, V = {}, {}
    for m in ("qwen", "gemma"):
        for l in open(f"{d}/{m}_solve.jsonl"):
            r = json.loads(l); S[(m, r["tid"], r["sample"])] = ans(r)
        for l in open(f"{d}/{m}_review.jsonl"):
            r = json.loads(l); V[(m, r["tid"], r["cond"])] = ans(r)
    return Y, S, V


# ---------------------------------------------------------------- J: exact tests
def sec_j():
    J = f"{R}/neutral-judge-20261121-a"; sets = json.load(open(f"{J}/sets.json")); P, row = {}, {}
    print("######## Study J, J1 (neutral - owner, accuracy on D): pre-registered bootstrap vs exact McNemar")
    pre = json.load(open(f"{J}/analysis.json"))
    for ts, cfg in sets.items():
        Y = {int(k): v for k, v in json.load(open(cfg["labels"])).items()}
        for m in ("gemma", "qwen"):
            V = {}
            for l in open(f"{J}/out/{ts}_{m}.jsonl"):
                r = json.loads(l); V[(r["tid"], r["cond"])] = ans(r)
            D = sorted({t for t, c in V})
            a = [V[(t, "neutral")] == Y[t] for t in D]; b = [V[(t, "owner")] == Y[t] for t in D]
            up, dn, p = mcnemar(a, b); P[f"{ts}:{m}"] = p; row[f"{ts}:{m}"] = (up, dn, len(D))
    H = holm(P)
    for k in P:
        up, dn, n = row[k]
        print(f"{k:12s} neutral-only right {up:3d}, owner-only right {dn:3d}, net {up-dn:+d}/{n} = {100*(up-dn)/n:+.1f} pp | "
              f"exact McNemar p={P[k]:.4f}, Holm={H[k]:.4f} | pre-registered bootstrap {f3(pre['J1 ' + k])}")


# ---------------------------------------------------------------- peer: decomposition, sources, exact tests
CELLS = (("01", "reader wrong, peer right", +1), ("10", "reader right, peer wrong", -1),
         ("00", "both wrong", +1), ("11", "both right", -1))


def cell(i, j): return f"{int(i)}{int(j)}"


def peer_rows(ts, m, src):
    """per problem: (cell, initial correct, self final correct, peer final correct) for reader m, peer source src."""
    Y, S, V = load_peer(ts)
    pm = m if src == "same" else OTHER[m]; ps = 1 if src == "same" else 0; cond = f"peer_{pm}"
    rows = []
    for t in sorted(Y):
        own, peer = S.get((m, t, 0)), S.get((pm, t, ps))
        if own is None or peer is None or (m, t, cond) not in V or (m, t, "self") not in V:
            continue
        rows.append((t, cell(own == Y[t], peer == Y[t]), own == Y[t], V[(m, t, "self")] == Y[t], V[(m, t, cond)] == Y[t]))
    return rows, len(Y)


def decompose(rows, final_idx, ref_idx):
    """Exact: mean(final) - mean(ref) = sum over cells of (count_c / N) * (rate of final correct - rate of ref correct).
    With ref = initial answer, ref is 1 in cells 1x and 0 in 0x, giving p01 a - p10 b + p00 c - p11 d."""
    N = len(rows); terms = {}
    for c, _, _ in CELLS:
        terms[c] = [(r[final_idx] - r[ref_idx]) if r[1] == c else 0.0 for r in rows]
    return N, terms


def sec_peer():
    print("\n######## Peer studies R/H/M by task x reader x peer source (post hoc)")
    out = {}
    for ts in TASKS:
        for m in ("gemma", "qwen"):
            for src in ("other", "same"):
                rows, N_all = peer_rows(ts, m, src)
                n = len(rows); idx = boot_idx(n)
                init = [r[2] for r in rows]; slf = [r[3] for r in rows]; pr = [r[4] for r in rows]
                key = f"{ts}:{m}:{src}"
                cnt = Counter(r[1] for r in rows)
                rate = {c: (sum(r[4] for r in rows if r[1] == c) / cnt[c] if cnt[c] else float("nan")) for c, _, _ in CELLS}
                rate_self = {c: (sum(r[3] for r in rows if r[1] == c) / cnt[c] if cnt[c] else float("nan")) for c, _, _ in CELLS}
                up, dn, p = mcnemar(pr, slf)
                print(f"\n== {key}: {n} of {N_all} problems with valid reader and peer first answers")
                print(f"   cells: " + ", ".join(f"{c} ({lab}) {cnt[c]}" for c, lab, _ in CELLS))
                print("   final-correct rate by cell, peer review: " + ", ".join(f"{c} {100*rate[c]:.1f}%" for c, _, _ in CELLS)
                      + " | self-review: " + ", ".join(f"{c} {100*rate_self[c]:.1f}%" for c, _, _ in CELLS))
                for lab, fi, ri in (("peer - initial", 4, 2), ("peer - self", 4, 3), ("self - initial", 3, 2)):
                    Nn, terms = decompose(rows, fi, ri)
                    tot = [sum(v) for v in zip(*terms.values())]
                    parts = "  ".join(f"{c}: {f3(boot(terms[c], idx))}" for c, _, _ in CELLS)
                    print(f"   {lab}: total {f3(boot(tot, idx))} = {parts}")
                two = (cnt["01"] * rate["01"] - cnt["10"] * (1 - rate["10"])) / n if cnt["01"] and cnt["10"] else float("nan")
                print(f"   two-term approximation of peer - initial: {100*two:+.1f} pp vs exact {100*(sum(pr)-sum(init))/n:+.1f} pp")
                print(f"   peer vs self, exact McNemar: peer-only right {up}, self-only right {dn}, p={p:.4f}")
                out[key] = {"n": n, "cells": dict(cnt), "rate_peer": rate, "rate_self": rate_self, "mcnemar_peer_self": [up, dn, p]}
            # source comparison on the SAME problems
            ro, _ = peer_rows(ts, m, "other"); rs, _ = peer_rows(ts, m, "same")
            O = {r[0]: r for r in ro}; Sx = {r[0]: r for r in rs}; common = sorted(set(O) & set(Sx))
            ab = [t for t in common if O[t][1] == "10" and Sx[t][1] == "10"]
            ac = [t for t in common if O[t][1] == "01" and Sx[t][1] == "01"]
            if ab:
                up, dn, p = mcnemar([not O[t][4] for t in ab], [not Sx[t][4] for t in ab])
                print(f"   [{ts}:{m}] abandon on the SAME {len(ab)} problems (reader right, both peers wrong): other-model peer "
                      f"{100*sum(not O[t][4] for t in ab)/len(ab):.1f}% vs same-model peer {100*sum(not Sx[t][4] for t in ab)/len(ab):.1f}% "
                      f"| discordant {up}/{dn}, McNemar p={p:.4f}")
            if ac:
                up, dn, p = mcnemar([O[t][4] for t in ac], [Sx[t][4] for t in ac])
                print(f"   [{ts}:{m}] accept on the SAME {len(ac)} problems (reader wrong, both peers right): other-model peer "
                      f"{100*sum(O[t][4] for t in ac)/len(ac):.1f}% vs same-model peer {100*sum(Sx[t][4] for t in ac)/len(ac):.1f}% "
                      f"| discordant {up}/{dn}, McNemar p={p:.4f}")
            # does the wrong peer share the reader's own wrong answer, etc.: answer agreement between same-model samples
    json.dump(out, open(f"{R}/revision-reanalysis-20261122/peer.json", "w"), indent=1, default=float)


# ---------------------------------------------------------------- transfer: consistent comparison, propagated uncertainty
def sec_transfer():
    print("\n######## Transfer of rates between tasks (post hoc). Prediction and observation are the SAME quantity:")
    print("         peer - initial answer on the target task. Predicted = sum_c p_c(target) * (rate_c(source) - initial_c).")
    print("         Uncertainty: source and target problems resampled independently (20,000 draws).")
    res = {}
    for m in ("gemma", "qwen"):
        for src in ("other", "same"):
            data = {ts: peer_rows(ts, m, src)[0] for ts in TASKS}
            for s_ts in TASKS:
                for t_ts in TASKS:
                    if s_ts == t_ts: continue
                    S_, T_ = data[s_ts], data[t_ts]
                    iS, iT = boot_idx(len(S_), seed=SEED + 7), boot_idx(len(T_), seed=SEED + 8)
                    cS = np.array([r[1] for r in S_]); fS = np.array([r[4] for r in S_], float)
                    cT = np.array([r[1] for r in T_]); fT = np.array([r[4] for r in T_], float); i0 = np.array([r[2] for r in T_], float)

                    def pred(ixS, ixT):
                        out = 0.0
                        for c, _, sign in CELLS:
                            mS = cS[ixS] == c; pc = np.mean(cT[ixT] == c)
                            rate = fS[ixS][mS].mean() if mS.any() else 0.0
                            init = 1.0 if c[0] == "1" else 0.0
                            out += pc * (rate - init)
                        return out
                    obs = fT.mean() - i0.mean(); pr = pred(np.arange(len(S_)), np.arange(len(T_)))
                    dif = np.sort([pred(iS[b], iT[b]) - (fT[iT[b]].mean() - i0[iT[b]].mean()) for b in range(0, B, 10)])  # 2,000 draws
                    lo, hi = dif[int(.025 * len(dif))], dif[int(.975 * len(dif)) - 1]
                    flag = "differs" if (lo > 0 or hi < 0) else "consistent"
                    res[f"{m}:{src}:{s_ts}->{t_ts}"] = [pr, obs, float(lo), float(hi)]
                    print(f"   {m} reads {src:5s} {s_ts:5s} -> {t_ts:5s}: predicted {100*pr:+.1f}, observed {100*obs:+.1f}, "
                          f"predicted - observed CI [{100*lo:+.1f}, {100*hi:+.1f}] {flag}")
    json.dump(res, open(f"{R}/revision-reanalysis-20261122/transfer.json", "w"), indent=1)


# ---------------------------------------------------------------- debate vs vote20: exact-style tests, tokens, seconds
def sec_debate():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hard_analyze import load, gemma_val
    from settings_analyze import vote
    from tokenizers import Tokenizer
    _t = Tokenizer.from_file("/home/lian/swarmworld/research/models/debate-or-vote/qwen2.5-7b-instruct-a09a354/tokenizer.json")
    class _Ids:
        def __init__(self, ids): self.input_ids = ids
    tok = lambda s: _Ids(_t.encode(s, add_special_tokens=False).ids)
    print("\n######## Debate vs 20-sample vote (Gemma-filled reader): sign-flip test, strict transitions, measured cost")
    G = {}
    for d in ("debate-hard-20261120-a", "debate-indep-20261121-a", "mmlupro-20261121-a"):
        for l in open(f"{R}/{d}/fill/gemma_out.jsonl"):
            r = json.loads(l); G[(d, r["id"])] = gemma_val(r["text"])
    sets = {"math": [("debate-hard-20261120-a", "out", ["H_full", "H_ans", "H_dovword", "H_vote20"]),
                     ("debate-indep-20261121-a", "out", ["H_full_indep", "H_ans_indep"])],
            "mmlu": [("mmlupro-20261121-a", "out-debate", ["H_full", "H_ans", "H_full_indep", "H_ans_indep", "H_vote20"])]}
    out = {}
    for ts, srcs in sets.items():
        A = {}
        for d, sub, arms in srcs:
            Gd = {k: v for (dd, k), v in G.items() if dd == d}
            LA = load(f"{R}/{d}/{sub}", Gd)
            for a in arms: A[a] = LA[a]
        D20, V20 = A["H_vote20"]; v20 = [vote(r[0], y) for y, r in V20]
        q = [ex["question"] for ex in D20["results"]]
        qtok = [len(tok(x).input_ids) for x in q]
        otok20 = sum(len(tok(t).input_ids) for ex in D20["results"] for t in ex["rounds"][0])
        cost20 = {"seconds": D20["seconds"], "output_tokens": otok20, "question_tokens_in": 20 * sum(qtok)}
        print(f"\n== {ts}: vote20 {100*sum(v20)/len(v20):.1f} | seconds {D20['seconds']} | output tokens {otok20:,} | question tokens in {20*sum(qtok):,}")
        for a in A:
            if a == "H_vote20": continue
            D, T = A[a]; K = len(T[0][1]) - 1
            vk = [vote(r[K], y) for y, r in T]; d = [x - y for x, y in zip(vk, v20)]
            win = sum(x == 1 and y == 0 for x, y in zip(vk, v20)); loss = sum(x == 0 and y == 1 for x, y in zip(vk, v20))
            otok = sum(len(tok(t).input_ids) for ex in D["results"] for rnd in ex["rounds"] for t in rnd)
            # input content tokens, excluding fixed template text: the question each call + (rounds >= 1) the 4 peers' previous outputs
            share = D["cfg"].get("share", "full")
            prev_tok = 0
            for ex, qt in zip(D["results"], qtok):
                L = [[len(tok(t).input_ids) for t in rnd] for rnd in ex["rounds"]]
                prev_tok += 5 * (K + 1) * qt
                if share == "full":
                    prev_tok += sum(sum(L[k - 1]) - L[k - 1][i] for k in range(1, K + 1) for i in range(5))
            p = signflip(d)
            out[f"{ts}:{a}"] = {"diff": boot(d), "p_signflip": p, "strict_wins": win, "strict_losses": loss,
                                "seconds": D["seconds"], "output_tokens": otok, "input_content_tokens": prev_tok, "vote20": cost20}
            print(f"   {a:13s} final - vote20 {f3(boot(d))} sign-flip p={p:.4f} | strict wins {win} / losses {loss} | "
                  f"seconds {D['seconds']} ({D['seconds']/D20['seconds']:.2f}x vote20) | output tokens {otok:,} ({otok/otok20:.2f}x) | "
                  f"input content tokens {prev_tok:,} ({prev_tok/(20*sum(qtok)):.1f}x)")
    json.dump(out, open(f"{R}/revision-reanalysis-20261122/debate.json", "w"), indent=1)


# ---------------------------------------------------------------- systems with a common denominator
def sec_system():
    from j_analyze import vote3
    J = f"{R}/neutral-judge-20261121-a"; sets = json.load(open(f"{J}/sets.json"))
    print("\n######## Two-model systems over ALL problems (common denominator; an invalid answer counts as wrong)")
    for ts, cfg in sets.items():
        Y, S, Vr = load_peer(ts); T = sorted(Y); N = len(T)
        VJ = {}
        for l in open(f"{J}/out/{ts}_gemma.jsonl"):
            r = json.loads(l); VJ[(r["tid"], r["cond"])] = ans(r)
        Dj = {t for t, c in VJ}
        g0 = [S.get(("gemma", t, 0)) == Y[t] for t in T]
        rows = {"Gemma s0": g0,
                "Gemma reads Qwen (owner)": [Vr.get(("gemma", t, "peer_qwen")) == Y[t] for t in T],
                "Qwen reads Gemma (owner)": [Vr.get(("qwen", t, "peer_gemma")) == Y[t] for t in T],
                "vote3 (g0, g1, q0)": [vote3(S.get(("gemma", t, 0)), S.get(("gemma", t, 1)), S.get(("qwen", t, 0))) == Y[t] for t in T],
                "Gemma judge, neutral, on D": [(VJ[(t, "neutral")] if t in Dj else S.get(("gemma", t, 0))) == Y[t] for t in T],
                "Gemma judge, owner, on D": [(VJ[(t, "owner")] if t in Dj else S.get(("gemma", t, 0))) == Y[t] for t in T],
                "oracle (g0 or q0)": [S.get(("gemma", t, 0)) == Y[t] or S.get(("qwen", t, 0)) == Y[t] for t in T]}
        idx = boot_idx(N)
        print(f"== {ts} (N = {N}; invalid first answers: gemma {sum(S.get(('gemma', t, 0)) is None for t in T)}, "
              f"qwen {sum(S.get(('qwen', t, 0)) is None for t in T)})")
        for k, v in rows.items():
            d = [float(a) - float(b) for a, b in zip(v, g0)]
            print(f"   {k:28s} {100*sum(v)/N:.1f}   - Gemma s0 {f3(boot(d, idx))}")


if __name__ == "__main__":
    os.makedirs(f"{R}/revision-reanalysis-20261122", exist_ok=True)
    sec = sys.argv[1]
    for name, fn in (("j", sec_j), ("peer", sec_peer), ("transfer", sec_transfer), ("system", sec_system), ("debate", sec_debate)):
        if sec in (name, "all"): fn()

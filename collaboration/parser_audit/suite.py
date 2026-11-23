"""Stress-test each codebase's parser on real debate reasoning + its OWN requested answer format, under
realistic perturbations. Target = the answer we append; a parser 'reads' correctly iff it returns it."""
import json, re, sys
from collections import defaultdict
import parsers as P

HIST = "/home/lian/swarmworld/research/reports/debate-vote-gsm8k-20261117-a/dense/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5.jsonl"
CUT = re.compile(r"\{\s*final answer|\\boxed|final answer|the answer is", re.I)

def bodies():
    out = []
    for line in open(HIST):
        d = json.loads(line)
        txt = list(d["0"]["responses"].values())[0]
        m = CUT.search(txt)
        body = (txt[: m.start()] if m else txt).strip()
        out.append((body, int(d["0"]["answer"])))
    return out

def variants(name, body, a):
    """Return {perturbation: (text, target)}."""
    b = a + 7
    big = 1000 * a + 250
    if name == "rec":
        def J(ans, reasoning=body, indent=None):
            return json.dumps({"reasoning": reasoning, "answer": str(ans), "confidence_level": 0.9}, indent=indent)
        return {"canonical": (J(a), a), "units_$": (J(f"${a}"), a), "commas": (J(f"{big:,}"), big),
                "decimal_.00": (J(f"{a}.00"), a), "newline_inside": (J(a, indent=2), a),
                "trailing_braces": (J(a) + " Note: half of {the total} was used.", a),
                "peer_after": (J(a) + ' Another agent answered {"answer": "%d"}.' % b, a),
                "peer_before": ('Another agent answered {"answer": "%d"}. My answer: ' % b + J(a), a),
                "loop_truncated": (((J(a) + " ") * 20)[:-3], a),
                "format_ignored": (body + f" The final answer is {a}.", a)}
    F = P.PARSERS[name]["fmt"]
    xml = F("X").startswith("<")
    def esc(s):
        return s.replace("<", "&lt;").replace(">", "&gt;") if xml else s.replace("{", "\\{").replace("}", "\\}")
    newline = F("\n%s\n" % a)
    return {"canonical": (f"{body} {F(a)}", a),
            "escaped_delims": (f"{body} {esc(F(a))}", a),
            "latex_text": (f"{body} {F('$' + str(a) + '$') if xml else F(chr(92) + 'text{' + str(a) + '}')}", a),
            "units_$": (f"{body} {F('$' + str(a))}", a),
            "commas": (f"{body} {F(f'{big:,}')}", big),
            "decimal_.00": (f"{body} {F(f'{a}.00')}", a),
            "newline_inside": (f"{body} {newline}", a),
            "trailing_braces": (f"{body} {F(a)} Here, \\frac{{1}}{{2}} of the total {{x}} was used.", a),
            "peer_after": (f"{body} {F(a)} (Another agent's answer was {F(b)}.)", a),
            "peer_before": (f"{body} Another agent's answer was {F(b)}. Rechecking my work, {F(a)}", a),
            "loop_truncated": (body + " " + ((F(a) + " ") * 20)[:-3], a),
            "format_ignored": (f"{body} The final answer is {a}.", a)}

def main():
    B = bodies()
    res = defaultdict(lambda: defaultdict(lambda: {"n": 0, "read": 0, "fail": 0, "misread": 0, "scored_correct": 0}))
    for name, spec in P.PARSERS.items():
        for body, a in B:
            for pert, (text, target) in variants(name, body, a).items():
                v = spec["fn"](text)
                c = res[name][pert]
                c["n"] += 1
                ok = v is not None and abs(v - target) < 1e-9
                c["read"] += ok
                c["fail"] += v is None
                c["misread"] += (v is not None and not ok)
                if v is None:
                    c["scored_correct"] += {"correct": 1, "wrong": 0, "answer_0": int(target == 0)}[spec["on_fail"]]
                else:
                    c["scored_correct"] += ok
    perts = list(next(iter(res.values())).keys())
    allp = sorted({p for r in res.values() for p in r}, key=lambda p: (p not in perts, perts.index(p) if p in perts else 0))
    print("read-correct % (fail % / misread %)  -- n =", len(B), "bodies per cell")
    print(f"{'perturbation':18s}" + "".join(f"{n:>20s}" for n in P.PARSERS))
    for p in allp:
        row = f"{p:18s}"
        for n in P.PARSERS:
            c = res[n].get(p)
            row += f"{'—':>20s}" if not c else f"{100*c['read']/c['n']:7.1f} ({100*c['fail']/c['n']:4.1f}/{100*c['misread']/c['n']:4.1f})"
        print(row)
    json.dump({n: dict(v) for n, v in res.items()}, open(sys.argv[1], "w"), indent=1)

if __name__ == "__main__":
    main()

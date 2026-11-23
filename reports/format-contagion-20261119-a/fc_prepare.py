"""Build the format-contagion twin experiment and the blind extraction audit (CPU, host)."""
import hashlib, json, random, re, sys
from pathlib import Path
import datasets
sys.path.insert(0, "/home/lian/swarmworld/research/reports/peer-theory-20261119")
from empties import original, robust  # noqa: E402

ROOT = Path("/home/lian/swarmworld/research")
VERSION = "format-contagion-v1"
DENSE = ROOT / "reports/debate-vote-gsm8k-20261117-a/dense/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5.jsonl"
ARROW = ROOT / "datasets/debate-or-vote/gsm8k-740312a/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow"
SUFFIX = ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
REPLICATES = 8
CONDITIONS = ("clean", "esc1", "esc4", "loop1", "loop4")
LOOP_REPEATS = 20
FINAL = re.compile(r"\{final answer:\s*([^{}]*?)\}")

def seed_for(*parts):
    return int(hashlib.sha256(json.dumps((VERSION,) + parts).encode()).hexdigest()[:8], 16)

def questions():
    """Debate-or-Vote order: test split, pandas sample(frac=1, random_state=0), head(300)."""
    df = datasets.Dataset.from_file(str(ARROW)).to_pandas()
    df = df.sample(frac=1, random_state=0).reset_index(drop=True).head(300)
    return list(df["question"]), [int(a.split("####")[-1].strip().replace(",", "")) for a in df["answer"]]

def break_format(resp, kind):
    """Rewrite only the last {final answer: X}: escaped braces, optionally followed by a repetition loop."""
    m = list(FINAL.finditer(resp))[-1]
    esc = "\\{final answer: " + m.group(1).strip() + "\\}"
    if kind == "esc":
        return resp[:m.start()] + esc + resp[m.end():]
    return resp[:m.start()] + esc + (" " + esc + " \\} \\)") * LOOP_REPEATS

def message(own, peers, question):
    """Debate-or-Vote dense round-1 message (src/main.py get_new_message), with the GSM8K suffix."""
    msg = "These are the recent opinions from other agents: "
    for p in peers:
        msg += f"\n\nOne of the agents' response: \n{p}\n"
    msg += f"\n\nThis was your most recent opinion:\n{own}\n"
    msg += f"\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{question}"
    return msg + SUFFIX

def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    qs, labels = questions()
    rows = [json.loads(l) for l in open(DENSE)]
    jobs, tasks, hidden, excluded = [], [], {}, []
    for ti, d in enumerate(rows):
        assert int(d["0"]["answer"]) == labels[ti], "question order mismatch"
        resps = list(d["0"]["responses"].values())
        # Eligible: all five round-0 responses parse with the published extractor and end their
        # stated answer in a single {final answer: X}, and their answers agree with the extracted value.
        if not all(original(r) is not None and FINAL.search(r) for r in resps):
            continue
        own, peers = resps[0], resps[1:]
        pick = seed_for(ti, "broken-peer") % 4
        variants = {"clean": peers,
                    "esc1": [break_format(p, "esc") if j == pick else p for j, p in enumerate(peers)],
                    "esc4": [break_format(p, "esc") for p in peers],
                    "loop1": [break_format(p, "loop") if j == pick else p for j, p in enumerate(peers)],
                    "loop4": [break_format(p, "loop") for p in peers]}
        # Manipulation check: the published extractor fails on every broken peer and robust still reads
        # the same answer. Tasks where a broken peer stays parseable (e.g. via a later \boxed{N}) are
        # excluded, so the contrast is exactly "peer answer readable vs not" with identical content.
        ok = all((original(p) is None) == (name != "clean" and (name.endswith("4") or j == pick))
                 and robust(p) == original(peers[j])
                 for name, ps in variants.items() for j, p in enumerate(ps))
        if not ok:
            excluded.append(ti)
            continue
        tasks.append(ti); hidden[ti] = labels[ti]
        for rep in range(REPLICATES):
            for name in CONDITIONS:
                jobs.append({"id": f"{ti}|{name}|{rep}", "prompt": message(own, variants[name], qs[ti]),
                             "seed": seed_for(ti, rep), "task": ti, "condition": name, "replicate": rep})
    random.Random(seed_for("order")).shuffle(jobs)
    with open(out / "contagion_jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    json.dump({"version": VERSION, "tasks": tasks, "excluded_manipulation_check": excluded, "jobs": len(jobs)}, open(out / "contagion_meta.json", "w"))
    json.dump(hidden, open(out / "labels_hidden.json", "w"))
    print("eligible tasks", len(tasks), "excluded by manipulation check", len(excluded), "jobs", len(jobs))

if __name__ == "__main__":
    main(sys.argv[1])

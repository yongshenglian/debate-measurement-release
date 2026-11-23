"""Study H (reports/debate-hard-20261120-a/PROTOCOL.md): settings_gen.py unchanged, plus three arms with a 2,048-token limit.
On MATH Levels 4-5 the 512/1,024 limits truncated 21-35% of solutions in the smoke run.
H_vote20 is the compute-matched baseline: 20 independent round-0 samples per problem (no debate), seeds seed(qi, 0, a)
for a = 0..19, so agents 0-4 share prompts and seeds with round 0 of the debate arms.
Usage: hard_gen.py <arm> <csv> <out.json> [limit]"""
import sys
import settings_gen as sg

sg.ARMS.update({
    "H_full":    dict(family="dmad", wording="dmad", chat=True, max_tokens=2048, share="full"),
    "H_ans":     dict(family="dmad", wording="dmad", chat=True, max_tokens=2048, share="answers"),
    "H_dovword": dict(family="dmad", wording="dov",  chat=True, max_tokens=2048, share="full"),
    "H_vote20":  dict(family="dmad", wording="dmad", chat=True, max_tokens=2048, share="full"),
})
if sys.argv[1:2] == ["H_vote20"]:
    sg.N_AGENTS, sg.N_DEBATE_ROUNDS = 20, 0

if __name__ == "__main__":
    sg.main()

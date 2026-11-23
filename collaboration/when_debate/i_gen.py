"""Study I (reports/debate-indep-20261121-a/PROTOCOL.md): study H debate arms plus the user's independence sentence.
settings_gen.py / hard_gen.py unchanged; the sentence is added as the last bullet of DMAD's debate "Instructions:".
Seeds as in study H, so round 0 is the same prompt and seed as H_full / H_ans.
Usage: i_gen.py <arm: H_full_indep | H_ans_indep> <csv> <out.json> [limit]"""
import hard_gen  # registers the H arms
import settings_gen as sg

INDEP = "- Take a fresh look at the work and use your own judgement; do not be misled by the other agents, however convincing they may sound.\n"
ANCHOR = "- Be concise and clear.\n"
sg.ARMS["H_full_indep"] = dict(sg.ARMS["H_full"])
sg.ARMS["H_ans_indep"] = dict(sg.ARMS["H_ans"])
_orig = sg.debate_msg


def debate_msg(cfg, q, texts, i):
    m = _orig(cfg, q, texts, i)
    assert m.count(ANCHOR) == 1
    return m.replace(ANCHOR, ANCHOR + INDEP)


sg.debate_msg = debate_msg

if __name__ == "__main__":
    sg.main()

"""Study M peer part: reverse_gen.py on MMLU-Pro. hard_reverse.py (2,048 tokens), the math task line replaced by a
multiple-choice one; with INDEP=1 also i_reverse.py's sentences. Same usage as reverse_gen.py."""
import os
import hard_reverse  # noqa: F401  (token limit)
if os.environ.get("INDEP") == "1":
    import i_reverse  # noqa: F401  (sentences)
import reverse_gen as rg

OLD = "Solve the following math problem step by step. The final answer is an integer.\n\nProblem:\n{q}"
NEW = "Answer the following multiple-choice question step by step.\n\nQuestion:\n{q}"
for name in ("TASK", "SELF", "PEER"):
    s = getattr(rg, name); assert s.startswith(OLD); setattr(rg, name, NEW + s[len(OLD):])

if __name__ == "__main__":
    rg.main()

"""Study H peer part: reverse_gen.py unchanged except the free-text limit, raised from 1,024 to 2,048 tokens
(the smoke run truncated 19% of Qwen and 35% of Gemma solutions at 1,024). Same usage as reverse_gen.py."""
import reverse_gen as rg

_SP = rg.SamplingParams
def _sp(*a, **k):
    if k.get("max_tokens") == 1024: k["max_tokens"] = 2048
    return _SP(*a, **k)
rg.SamplingParams = _sp

if __name__ == "__main__":
    rg.main()

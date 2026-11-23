# Re-analysis of existing outputs for the paper revision (2026-11-22)

- **Prompt:** an external review of the TMLR draft. The user approved the revision plan ("approave all").
- **Scope:** no new generation; this note covers phase 1 only.
- **Status:** every analysis here is **post hoc** relative to the original protocols.
- **Script:** `collaboration/when_debate/revision_reanalysis.py <j|peer|transfer|system|debate>`.
- **Outputs:** `*.out` and `*.json` in this folder.
- **Resampling:** always by problem.

## 1. Study J's significant result is test-sensitive (`j` section; printed in chat, reproduced by the script)

- **Gemma on MMLU-Pro, neutral − owner:** 29 problems fixed and 12 broken; net +17/250 = +6.8 pp.
  - Exact McNemar p = .0115, which is **.069 after Holm** over the six J1 tests.
  - The pre-registered bootstrap Holm p was .047.
- **Reading:** the effect is suggestive, not established. Study K is the pre-registered replication.

## 2. Exact four-term decomposition (`peer.out`)

- **The identity:** Δ = p01·a − p10·b + p00·c − p11·d, relative to the first answer.
  - It is exact, and it is reported both relative to the first answer and relative to self-review.
- **How good the two-term approximation is:** within 0.8 pp of the exact change in every task × reader × source cell (largest gaps: −2.0 vs −1.3 and +0.7 vs +1.5).
- **Both-wrong recovery (c):** 0–5%. Its term contributes at most +1.2 pp.
- **Self-review moves the p00 and p11 terms on MMLU-Pro** (Gemma: −1.5 and +1.3 pp).
  - So peer − self and peer − first answer differ there by up to 1.3 pp.

## 3. Peer source matters on knowledge questions (`peer.out`, matched problems)

- **Same problems, different peer:** the 24 MMLU-Pro problems where Gemma was right and *both* peers were wrong.
  - Gemma abandoned its answer **87.5%** of the time with its own second sample as peer, vs **41.7%** with Qwen as peer.
  - Discordant 12 vs 1, McNemar p = .003.
- **The same direction on MATH, not significant:** 36.8% vs 21.1%, n = 19.
- **Reading:** the pooled 42% Gemma MMLU-Pro abandon rate in the draft mixes two very different peer sources.
  - On identical problems, a wrong solution written by the reader's own model persuades it about twice as often as one written by the other model.
  - This is post hoc and based on n = 24, so it needs a pre-registered test.

## 4. Transfer of rates, with the same quantity on both sides (`transfer.out`)

- **The draft's error:** the pre-registered HR3 test compared a prediction of *peer − first answer* with an interval for *peer − self-review*, and it ignored uncertainty in the source rates.
- **The fix:** predicted and observed are both peer − first answer on the target task, and source and target problems are bootstrapped independently.
- **GSM8K → MATH, Gemma reading Qwen:** predicted −3.7, observed −1.3, (predicted − observed) CI [−5.7, +0.7].
  - **Consistent.** The draft's "transfer fails" claim for this pair does not hold.
- **Where transfer does fail:** Gemma, in every pair involving MMLU-Pro (e.g. MATH → MMLU-Pro: predicted +2.3, observed −2.3).
  - Also Qwen reading Gemma, MMLU-Pro → MATH.
- **Where it holds:** most Qwen transfers.
- **Reading:** rates transfer between the two math tasks but not between math and knowledge questions for the strong reader.

## 5. Systems with a common denominator (`system.out`)

- **The rule:** all problems, with an invalid answer counted as wrong.
  - Only one invalid first answer exists (Qwen, MATH), so the numbers barely change.
- **MMLU-Pro:**

  | System | vs Gemma s0 |
  |---|---|
  | vote3 | +1.2 [−0.7, +3.0] |
  | neutral judge | +0.2 [−2.0, +2.3] |
  | owner judge | −2.7 [−5.0, −0.3] |
  | oracle | +7.7 |

## 6. Debate vs the 20-vote: tests and measured cost (`debate.out`)

- **Tests:** the sign-flip randomisation tests agree with the pre-registered bootstrap intervals.
  - MATH: every arm below the 20-vote, p = .0003–.03.
  - MMLU-Pro: full reasoning p ≈ .06; answers-only −1.0 [−4.8, +2.8], p = .61, which is **inconclusive, not a tie**.
- **Strict transitions:** problems the debate gets right and the 20-vote gets wrong, and vice versa.
  - MATH: 6–8 vs 15–25.
  - MMLU-Pro: 13–21 vs 14–35.
- **Measured cost at equal calls**, same GPU and harness:

  | Measure | Debate vs 20-vote |
  |---|---|
  | Wall-clock | **1.5–4.0×** |
  | Output tokens | 0.77–1.00× |
  | Input content tokens, full reasoning (excluding fixed template text) | 7–27× |
  | Input content tokens, answers-only | about 1× |

- **Reading:** "equal calls" is not equal compute.
  - Full-reasoning debate costs far more input tokens and time.
  - Answers-only debate costs about the same tokens but 1.5–1.7× the time.
- **The draft said** "token matching would favour voting further". That holds for input tokens and time, not for output tokens.

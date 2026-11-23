# Draft: email to Wynn et al. (2509.05396, "Talk Isn't Always Cheap"). NOT SENT: the user sends it from their own account.

**Subject:** Talk Isn't Always Cheap: replication of the open-model GSM8K configurations, and a tie-break question

Dear authors,

We are auditing measurement pipelines in multi-agent debate papers, and we re-ran your open-model GSM8K configurations with your released code (TheNormativityLab/talk-aint-cheap @47ffdb5).
- **Setup:** 100 problems × 5 seeds, Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.2 served with vLLM, your prompts and parameters.
- Thank you for releasing it. Three things we'd like to check with you:

1. **Missing configs.** `src/conf/basic_debate.yaml` refers to `llm_conf` entries `vec_llama_3_1_8B` and `vec_mistral_7B`, which are not in the repository.
   - We recreated them, pointing to local vLLM servers.
   - Could you share the originals (sampling settings in particular)?
2. **Tie-breaking in `most_frequent`** (`src/analysis/evaluation.py:87`). A three-way tie goes to the first answer in the list, which is llm1's.
   - In the 1 Llama + 2 Mistral configuration, llm1 is Llama, the strongest agent. Ties are frequent at round 0 and become rare after debate, so this implicit advantage shrinks over rounds.
   - **Effect:** with your rule, the vote changes by −13.0 pp [−17.2, −8.8] from round 0 to round 2. With ties split evenly, it changes by +0.6 pp [−2.7, +4.0].
   - Was first-agent tie-breaking intended? And was the agent order in the paper's runs the same as in `basic_debate.yaml`?
3. **Replication summary:**
   - Your answer parsing is sound: re-scoring with a format-tolerant extractor gives the same results.
   - The 2 Llama + 1 Mistral decline replicates: −8.4 pp [−11.8, −5.2].
   - The 3×Llama (−1.4 [−3.4, +0.6]) and 3×Mistral (+0.8 [−2.6, +4.2]) changes were not significant in our runs. The paper reports −3.4 and +2.8.

We are happy to share all raw outputs and scripts. We'd like to describe your setup accurately in our write-up.

Best regards,
[name, affiliation]

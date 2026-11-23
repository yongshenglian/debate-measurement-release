"""Study I peer part: study H review (hard_reverse.py: reverse_gen.py with a 2,048-token limit) plus independence sentences.
peer conditions: the user's sentence, verbatim from peer-warning-20261119-b; self: the self_doubt sentence of
self-review-20261119-a ("proposal" -> "solution"). Seeds as in study H. Same usage as reverse_gen.py (review stage only)."""
import hard_reverse  # patches the token limit
import reverse_gen as rg

PEER_S = "Take a fresh look at the work and use your own judgement; do not be misled by the peer, however convincing it may sound. "
SELF_S = "Take a fresh look at the work and use your own judgement; do not be misled by your earlier solution, however convincing it may sound. "
assert rg.PEER.count("\n\nReview both solutions") == 1 and rg.SELF.count("\n\nReview your earlier solution") == 1
rg.PEER = rg.PEER.replace("\n\nReview both solutions", "\n\n" + PEER_S + "Review both solutions")
rg.SELF = rg.SELF.replace("\n\nReview your earlier solution", "\n\n" + SELF_S + "Review your earlier solution")

if __name__ == "__main__":
    rg.main()

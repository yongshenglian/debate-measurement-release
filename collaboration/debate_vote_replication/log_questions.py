"""Extract top-level benchmark questions from Docker logs containing tqdm CR output."""

QUESTION_PREFIX = "Question:  "
QUESTION_SUFFIX = (
    ' Make sure to state your final answer in curly brackets at the very end of '
    'your response, just like: "{final answer: 123}".'
)


def questions_from_log(text):
    questions = []
    for line in text.splitlines():
        marker = line.find(QUESTION_PREFIX)
        if marker < 0:
            continue
        rendered = line[marker + len(QUESTION_PREFIX):].rstrip()
        if not rendered.endswith(QUESTION_SUFFIX):
            # A response may mention the word Question; only the benchmark's exact
            # instruction suffix identifies a top-level question record.
            continue
        questions.append(rendered[:-len(QUESTION_SUFFIX)])
    return questions

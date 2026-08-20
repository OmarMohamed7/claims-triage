# Shared between scripts/build_policy_index.py and
# src/agents/policy_retrieval.py so BM25 tokenizes corpus and query the
# same way.
STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have he in is it its of on
    that the to was were will with i my me we our you your she they
    them this these those but or not no so if than then there here
    when where which who whom what how why do does did done having
    been being am during into over under again further once up down
    out off above below between through before after about against
    all any both each few more most other some such only own same too
    very can just should now
    """.split()
)

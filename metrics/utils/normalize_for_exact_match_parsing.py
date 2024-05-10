import re

def _replace_implicit_joins(sql_query: str) -> str:
    """Replace ',' in the FROM clause with 'join'."""
    return re.sub(r'FROM\s+((?:\w+\s+AS\s+\w+|\w+)(?:\s*,\s*(?:\w+\s+AS\s+\w+|\w+))*)',
                  repl=lambda match: match.group(0).replace(',', ' JOIN '),
                  string=sql_query, flags=re.IGNORECASE)


def _replace_inner_join(sql_query: str) -> str:
    """Replaces the keyword 'inner join' with the keyword 'join' since the inner join is the default join method."""
    return re.sub(r' INNER JOIN ', repl=" JOIN ", string=sql_query, flags=re.IGNORECASE)


def _normalize_unequality_operator(sql_query: str) -> str:
    return re.sub(r" <> ", repl=" != ", string=sql_query)


def _replace_backquote(sql_query: str) -> str:
    return sql_query.replace("`", "'")


def _find_closing_parenthesis_pair(open_parenthesis_end: int, sql_query: str) -> int:
    right_parenthesis_start = None
    for right_parenthesis_match in re.finditer('\)', sql_query[open_parenthesis_end:]):
        right_parenthesis_start = right_parenthesis_match.start() + open_parenthesis_end
        # Check if this is the corresponding right parenthesis
        substring = sql_query[open_parenthesis_end: right_parenthesis_start]
        if substring.count("(") == substring.count(")"):
            return right_parenthesis_start
    return right_parenthesis_start


def _remove_parenthesis_pair(text: str, open_parenthesis_end: int, closing_parenthesis_start: int) -> str:
    return text[:open_parenthesis_end-1] + text[open_parenthesis_end:closing_parenthesis_start] + \
        text[closing_parenthesis_start + 1:]


def _remove_redundant_parentheses_in_where_clauses(sql_query: str) -> str:
    where_match = re.search("WHERE \(", sql_query, re.IGNORECASE)
    while where_match is not None:
        closing_parenthesis_start = _find_closing_parenthesis_pair(where_match.end(), sql_query)
        sql_query = _remove_parenthesis_pair(sql_query, where_match.end(), closing_parenthesis_start)
        where_match = re.search("WHERE \(", sql_query, re.IGNORECASE)

    return sql_query


def normalize_for_exact_match_parsing(sql_query: str) -> str:
    """Normalizes the sql query to be parsable by the parser of exact match"""

    sql_query = _replace_implicit_joins(sql_query)

    sql_query = _replace_inner_join(sql_query)

    sql_query = _normalize_unequality_operator(sql_query)

    sql_query = _replace_backquote(sql_query)

    sql_query = _remove_redundant_parentheses_in_where_clauses(sql_query)

    return sql_query

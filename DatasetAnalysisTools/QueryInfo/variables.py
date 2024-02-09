from typing import Union, Any, Literal
from abc import ABC


class Variable(ABC):
    """
    Represents a variable of a SQL query.
    """
    pass

    @classmethod
    def is_query(cls) -> bool:
        """
        Returns if the class can represent a query.
        """
        return False


class Table(Variable):

    def __init__(self, name: str, alias: str = None):
        self.name = name.lower()
        self.alias = alias.lower() if alias is not None else None

    def aliases(self, shallow_search: bool = False) -> dict[str, str]:
        """
        Returns a dictionary with the aliases and the corresponding schema element names. If the alias is not for
        a schema element the value is None.
        """
        return {self.alias: self.name} if self.alias is not None else {}


class QueryAsTable(Variable):

    def __init__(self, query: Union['QueryInfo', 'SetOperator'], alias: str = None):
        self.query = query
        self.alias = alias.lower() if alias is not None else None

    @classmethod
    def is_query(cls) -> bool:
        """ Returns if the class can represent a query. """
        return True

    def tables(self, shallow_search: bool = False, unique: bool = False):
        """
        Returns the tables of the schema existing in the query.
        """
        return self.query.tables(shallow_search=shallow_search, unique=unique)

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, 'Column']]:
        """
        Returns the columns of the schema existing in the query.
        """
        return self.query.columns(return_format=return_format, shallow_search=shallow_search, unique=unique)

    def values(self, shallow_search: bool = False):
        """ Returns the values existing in the query. """
        return self.query.values(shallow_search=shallow_search)

    def aliases(self, shallow_search: bool = False) -> dict[str, str]:
        """
        Returns a dictionary with the aliases and the corresponding schema element names. If the alias is not for
        a schema element the value is None.
        """
        als = {self.alias: None} if self.alias is not None else {}

        als.update(self.query.aliases(shallow_search=shallow_search))

        return als

    def subqueries(self, shallow_search: bool = True) -> list['QueryInfo']:
        """ Returns the query or the subqueries of a set operator. """
        if self.query.is_query():
            return [self.query]
        else:
            return self.query.subqueries(shallow_search=shallow_search)

    def setOperators(self, shallow_search: bool = False) -> list['SetOperators']:
        """ Returns the query if it is a set operator else an empty list. """
        return [self.query] if hasattr(self.query, "name") and self.query.name() == "set_operator" else []

    def _get_property_list(self, property_name: str, **kwargs) -> list[str]:
        return getattr(self.query, property_name)(**kwargs)

    def joins(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("joins", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def aggregates(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("aggregates", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def comparison_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("comparison_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def logical_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("logical_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def like_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("like_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def arithmetic_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("arithmetic_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def membership_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("membership_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def null_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("null_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})


class Column(Variable):
    def __init__(self, name: str, table_name: str = None, alias: str = None):
        """
        Initializes the information about a column.
        Args:
            name (str): The name of the column.
            table_name (str): The table, in which the column exists. It can be the real name of a table or an
                              alias of a table
            alias (str): The alias of the column
        """
        self.name = name.lower()
        self.table_name = table_name.lower() if table_name is not None else None
        self.alias = alias.lower() if alias is not None else None

    @property
    def fullname(self) -> str:
        return f"{f'{self.table_name}.' if self.table_name is not None else ''}{self.name}"

    def aliases(self, shallow_search: bool = False) -> dict[str, str]:
        """
        Returns a dictionary with the aliases and the corresponding schema element names. If the alias is not for
        a schema element the value is None.
        """
        return {self.alias: self.name} if self.alias is not None else {}


class OperatorAsColumn(Variable):

    def __init__(self, operator: 'Operator', alias: str = None):
        """
        Initializes the information of an operation that appears as column. E.g., select count(*) from table
        Args:
            operator (Operator): The operator that acts as a column
            alias (str)
        """
        self._operator = operator
        self.alias = alias.lower() if alias is not None else None

    def tables(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        """
        Returns the tables of the schema existing in the operator.
        """
        return self._operator.tables(shallow_search=shallow_search, unique=unique)

    def values(self, shallow_search: bool = False) -> list[Any]:
        """ Returns the values existing in the operator. """
        return self._operator.values(shallow_search=shallow_search)

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """
        Returns the columns of the schema existing in the operator.
        """
        return self._operator.columns(return_format=return_format, shallow_search=shallow_search, unique=unique)

    def aliases(self, shallow_search: bool = False) -> dict[str, str]:
        """
        Returns a dictionary with the aliases and the corresponding schema element names. If the alias is not for
        a schema element the value is None.
        """
        als = {self.alias: None} if self.alias is not None else {}

        als.update(self._operator.aliases(shallow_search=shallow_search))

        return als

    def subqueries(self, shallow_search: bool = True) -> list['QueryInfo']:
        """
        Returns a list of subqueries existing in the operator.
        """
        return self._get_property_list("subqueries", **{"shallow_search": shallow_search})

    def setOperators(self, shallow_search: bool = False) -> list['SetOperator']:
        """ Returns a list of setOperators existing in the operator. """
        return self._get_property_list("setOperators")

    def _get_property_list(self, property_name: str, **kwargs) -> list[Union[str, 'QueryInfo', 'SetOperator']]:
        return getattr(self._operator, property_name)(**kwargs)

    def joins(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("joins", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def aggregates(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("aggregates", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def comparison_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("comparison_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def logical_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("logical_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def like_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("like_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def arithmetic_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("arithmetic_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def membership_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("membership_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def null_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, 'Operator']:
        return self._get_property_list("null_operators", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})


class Value(Variable):
    def __init__(self, name: Union[str, int, float]):
        self.name = name

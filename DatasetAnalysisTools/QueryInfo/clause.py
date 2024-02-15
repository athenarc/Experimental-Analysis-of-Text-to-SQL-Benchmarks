from abc import ABC, abstractmethod
from typing import Union, Any, Literal

from DatasetAnalysisTools.QueryInfo.operators import Join, MembershipOperator, NullOperator, ComparisonOperator, \
    LogicalOperator, LikeOperator, ArithmeticOperator, Aggregate, Operator
from DatasetAnalysisTools.QueryInfo.variables import Column, Table, QueryAsTable, OperatorAsColumn


class Clause(ABC):

    @classmethod
    def is_query(cls) -> bool:
        """ Returns if the class can represent a query. """
        return False

    @abstractmethod
    def _get_property_list(self, property_name: str, **kwargs) -> list[str]:
        pass

    @abstractmethod
    def _get_property_dict(self, property_name: str, **kwrags) -> dict:
        pass

    def tables(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        """ Returns the tables existing in a clause. """
        return self._get_property_list("tables", **{"shallow_search": shallow_search,
                                                    "unique": unique})

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """ Returns the columns existing in a clause. """
        return self._get_property_list("columns", **{"shallow_search": shallow_search,
                                                     "return_format": return_format, "unique": unique})

    def values(self, shallow_search: bool = False) -> list[Any]:
        """ Returns the values existing in a clause. """
        return self._get_property_list("values", **{"shallow_search": shallow_search})

    def aliases(self, shallow_search: bool = False) -> dict[str, str]:
        """ Returns the aliases existing in a clause. """
        return self._get_property_dict("aliases", **{"shallow_search": shallow_search})

    def joins(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        """
        Returns the joins existing in a clause.

        Args:
            shallow_search (bool): If True the joins of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("joins", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def aggregates(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        """
        Returns the aggregates existing in a clause.

        Args:
            shallow_search (bool): If True the aggregates of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("aggregates", **{"shallow_search": shallow_search,
                                                        "return_format": return_format})

    def comparison_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str]:
        """
        Returns the comparison operators existing in a clause.

        Args:
            shallow_search (bool): If True the comparison operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("comparison_operators", **{"shallow_search": shallow_search,
                                                                  "return_format": return_format})

    def logical_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[
        str]:
        """
        Returns the logical operators existing in a clause.

        Args:
            shallow_search (bool): If True the logical operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("logical_operators", **{"shallow_search": shallow_search,
                                                               "return_format": return_format})

    def like_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        """
        Returns the like operators existing in a clause.

        Args:
            shallow_search (bool): If True the like operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("like_operators", **{"shallow_search": shallow_search,
                                                            "return_format": return_format})

    def arithmetic_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str]:
        """
        Returns the arithmetic operators existing in a clause.

        Args:
            shallow_search (bool): If True the arithmetic operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("arithmetic_operators", **{"shallow_search": shallow_search,
                                                                  "return_format": return_format})

    def membership_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str]:
        """
        Returns the membership operators existing in a clause.

        Args:
            shallow_search (bool): If True the membership of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("membership_operators", **{"shallow_search": shallow_search,
                                                                  "return_format": return_format})

    def null_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        """
        Returns the null operators existing in a clause.

        Args:
            shallow_search (bool): If True the null operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("null_operators", **{"shallow_search": shallow_search,
                                                            "return_format": return_format})

    def operatorTypes(self, shallow_search: bool = False, with_names: bool = False) \
            -> list[str]:
        """
        Returns a list with the operatorTypes existing in the clause.
        Args:
            shallow_search (bool): If True the operator types of the subqueries will not be considered.
            with_names (bool): If True the operator types are returned with their names, else with their
                abbreviations.
        """
        opTypes = []
        for operatorType in [Join, Aggregate, ComparisonOperator, LogicalOperator, LikeOperator,
                             ArithmeticOperator, MembershipOperator, NullOperator]:
            opTypes.extend([operatorType.class_name() if with_names else operatorType.abbr()
                            for _ in range(len(getattr(self, f"{operatorType.class_name()}s")()))])

        # If operator types of subqueries must not be considered
        if shallow_search and hasattr(self, "subqueries"):
            # Get the operator types of the subqueries
            for subquery in self.subqueries():
                subquery_opTypes = subquery.operatorTypes(with_names=with_names)
                # Remove the operator types of the subquery
                for subquery_opType in subquery_opTypes:
                    opTypes.remove(subquery_opType)

        return opTypes

    def operators(self, shallow_search: bool = False) -> list[str]:
        """
        Returns a list with the operators existing in the clause.
        Args:
            shallow_search (bool): If True the operator types of the subqueries will not be considered.
        """
        ops = []
        for operatorType in [Join, Aggregate, ComparisonOperator, LogicalOperator, LikeOperator,
                             ArithmeticOperator, MembershipOperator, NullOperator]:
            ops.extend(getattr(self, f"{operatorType.class_name()}s")())

        # If the operators of the subqueries must not be considered
        if shallow_search and hasattr(self, "subqueries"):
            # Get the operators of the subqueries
            for subquery in self.subqueries():
                subquery_ops = subquery.operators()
                # Remove the operators of the subquery
                for subquery_op in subquery_ops:
                    ops.remove(subquery_op)
        return ops


class SelectClause(Clause):

    def __init__(self, attributes: list[Union[Column, OperatorAsColumn, 'QueryInfo', 'SetOperator', str]],
                 distinct: bool):
        self._attributes = attributes
        self.distinct = distinct

    @classmethod
    def class_name(cls) -> str:
        return "select"

    @classmethod
    def abbr(cls) -> str:
        return "S"

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """
        Returns the columns of the schema existing in the select clause or in any nested query in the select clause
        """
        cols = []
        for attribute in self._attributes:
            attribute_type = type(attribute)
            if attribute_type is str:
                continue
            if attribute_type is Column:
                if return_format == "name":
                    cols.append(attribute.name)
                elif return_format == "complete_name":
                    cols.append(attribute.fullname)
                else:  # return_format == "raw"
                    cols.append(attribute)
            else:
                # If the attribute is not a subquery while the shallow search is enabled
                if not (shallow_search and attribute.is_query()):
                    cols.extend(attribute.columns(return_format=return_format, shallow_search=shallow_search,
                                                  unique=unique))
        return cols

    def subqueries(self, shallow_search: bool = True) -> list['QueryInfo']:
        """
        Returns a list with the subqueries existing in the select clause.
        """
        subqs = []
        for attribute in self._attributes:
            if hasattr(attribute, "is_query") and attribute.is_query():
                subqs.append(attribute)
            elif hasattr(attribute, "subqueries"):
                subqs.extend(attribute.subqueries())
        return subqs

    def setOperators(self, shallow_search: bool = False) -> list['SetOperator']:
        setOps = []
        for attribute in self._attributes:
            try:
                if attribute.class_name() == "set_operator":
                    setOps.append(attribute)
            except (AttributeError, TypeError):
                continue
        return setOps

    def _get_property_list(self, property_name: str, **kwargs):
        property_values = []
        for attribute in self._attributes:
            # If the attribute is not a subquery while the shallow search is enabled
            if not ("shallow_search" in kwargs and kwargs["shallow_search"] and hasattr(attribute, "is_query") and
                    attribute.is_query()):
                if hasattr(attribute, property_name):
                    property_values.extend(getattr(attribute, property_name)(**kwargs))
        return property_values

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        property_values = {}
        for attribute in self._attributes:
            # If the attribute is not a subquery while the shallow search is enabled
            if not ("shallow_search" in kwargs and kwargs["shallow_search"] and hasattr(attribute, "is_query") and
                    attribute.is_query()):
                if hasattr(attribute, property_name):
                    property_values.update(getattr(attribute, property_name)(**kwargs))
        return property_values

    def columns_num(self) -> int:
        """
        Returns the number of columns in the select clause. Considered columns may be references of schema elements or
        an operator as count(*).
        """
        return len(self._attributes)


class FromClause(Clause):

    def __init__(self, tables: list[Union[QueryAsTable, Table, Join]]):
        self._tables = tables

    @classmethod
    def class_name(cls) -> str:
        return "from"

    @classmethod
    def abbr(cls) -> str:
        return "F"

    def tables(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        """
        Returns the tables of the schema existing in the from clause or in any nested query in the from clause
        """
        schema_tables = []
        for table in self._tables:
            if type(table) is Table:
                schema_tables.append(table.name)
            else:
                if not (shallow_search and table.is_query()):
                    schema_tables.extend(table.tables(shallow_search=shallow_search, unique=unique))
        return schema_tables

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """
        Returns the columns of the schema existing in the in any nested query in the from clause
        """
        cols = []
        for table in self._tables:
            if type(table) is not Table:
                # If the table is not a query while shallow search is enabled
                if not (shallow_search and table.is_query()):
                    cols.extend(table.columns(return_format=return_format, shallow_search=shallow_search,
                                              unique=unique))
        return cols

    def subqueries(self, shallow_search: bool = True) -> list['QueryInfo']:
        """ Returns a list with the subqueries existing in the from clause. """
        subqs = []
        for table in self._tables:
            if type(table) is not Table:
                subqs.extend(table.subqueries())
        return subqs

    def setOperators(self, shallow_search: bool = False) -> list['SetOperator']:
        """ Returns a list with the set operators existing in the from clause. """
        return self._get_property_list("setOperators")

    def _get_property_list(self, property_name: str, **kwargs) -> \
            list[Union[str, 'SetOperator']]:
        property_values = []
        for table in self._tables:
            # If the table is not a subquery while the shallow search is enabled
            if not ("shallow_search" in kwargs and kwargs["shallow_search"] and table.is_query()):
                if hasattr(table, property_name):
                    property_values.extend(getattr(table, property_name)(**kwargs))
        return property_values

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        property_values = {}
        for table in self._tables:
            # If the table is not a subquery while the shallow search is enabled
            if not ("shallow_search" in kwargs and kwargs["shallow_search"] and table.is_query()):
                if hasattr(table, property_name):
                    property_values.update(getattr(table, property_name)(**kwargs))
        return property_values


class WhereClause(Clause):

    def __init__(self, condition: Union[ComparisonOperator, LogicalOperator, LikeOperator, ArithmeticOperator,
    MembershipOperator, NullOperator]):
        self._condition = condition

    @classmethod
    def class_name(cls) -> str:
        return "where"

    @classmethod
    def abbr(cls) -> str:
        return "W"

    def subqueries(self) -> list['QueryInfo']:
        """ Returns a list with the subqueries existing in the where clause. """
        return self._get_property_list("subqueries")

    def setOperators(self, shallow_search: bool = False) -> list['SetOperator']:
        """ Returns a list with the set operators existing in the where clause. """
        return self._get_property_list("setOperators")

    def _get_property_list(self, property_name: str, **kwargs) -> \
            list[Union[str, 'QueryInfo', 'SetOperator']]:
        return getattr(self._condition, property_name)(**kwargs) \
            if hasattr(self._condition, property_name) else []

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        return getattr(self._condition, property_name)(**kwargs) \
            if hasattr(self._condition, property_name) else {}

    @staticmethod
    def _get_subconditions(conditions: list[Operator]) -> list[Operator]:
        subconditions = []
        for condition in conditions:
            for operand in condition.operands:
                if isinstance(operand, Operator):
                    subconditions.append(operand)
        return subconditions

    def conditions_num(self):
        """ Returns the number of conditions in the where clause. """
        conditions = [[self._condition]]

        while len(conditions[-1]) > 0:
            conditions.append(self._get_subconditions(conditions[-1]))

        # Remove the last entry which has 0 conditions
        conditions = conditions[:-1]

        # Return the number of all conditions
        return sum([len(subconds) for subconds in conditions])


class GroupByClause(Clause):

    def __init__(self, columns: list[Column]):

        # Verify the type of the given columns
        for column in columns:
            if type(column) is not Column:
                raise TypeError("Group by clause can contain only columns!")

        self._columns = columns

    @classmethod
    def class_name(cls) -> str:
        return "groupby"

    @classmethod
    def abbr(cls) -> str:
        return "G"

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """
        Returns the columns of the schema existing in the groupby clause.
        """
        if return_format == "name":
            return [column.name for column in self._columns]
        elif return_format == "complete_name":
            return [column.fullname for column in self._columns]
        else:  # return_format == "raw
            return self._columns

    def _get_property_list(self, property_name: str, **kwargs) -> list[str]:
        property_values = []
        for column in self._columns:
            if hasattr(column, property_name):
                property_values.extend(getattr(column, property_name)(**kwargs))

        return property_values

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        property_values = {}
        for column in self._columns:
            if hasattr(column, property_name):
                property_values.update(getattr(column, property_name)(**kwargs))
        return property_values


class HavingClause(Clause):

    def __init__(self, condition: Union[Aggregate, ComparisonOperator, LogicalOperator, LikeOperator,
                 ArithmeticOperator, MembershipOperator, NullOperator]):
        self._condition = condition

    @classmethod
    def class_name(cls) -> str:
        return "having"

    @classmethod
    def abbr(cls) -> str:
        return "H"

    def subqueries(self) -> list['QueryInfo']:
        """
        Returns the subqueries existing in the having clause.
        """
        return self._get_property_list("subqueries")

    def setOperators(self, shallow_search: bool = False) -> list['SetOperator']:
        """ Returns a list with the set operators existing in the having clause. """
        return self._get_property_list("setOperators")

    def _get_property_list(self, property_name: str, **kwargs) \
            -> list[Union[str, 'QueryInfo', 'SetOperator']]:
        return getattr(self._condition, property_name)(**kwargs) \
            if hasattr(self._condition, property_name) else []

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        return getattr(self._condition, property_name)(**kwargs) \
            if hasattr(self._condition, property_name) else {}


class OrderByClause(Clause):

    def __init__(self, columns: list[OperatorAsColumn, Column], orders: list[str]):
        self._columns = columns
        self._orders = orders

    @classmethod
    def class_name(cls) -> str:
        return "orderby"

    @classmethod
    def abbr(cls) -> str:
        return "O"

    def orders(self) -> list[str]:
        """Returns the orderings of the columns in the clause"""
        return self._orders

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """
        Returns the columns of the schema existing in the orderby clause.
        """
        returned_columns = []
        for column in self._columns:
            if isinstance(column, Column):
                if return_format == "name":
                    returned_columns.append(column.name)
                elif return_format == "complete_name":
                    returned_columns.append(column.fullname)
                else:  # return_format == "raw"
                    returned_columns.append(column)
        return returned_columns

    def _get_property_list(self, property_name: str, **kwargs) -> list[str]:
        property_values = []
        for column in self._columns:
            if hasattr(column, property_name):
                property_values.extend(getattr(column, property_name)(**kwargs))
        return property_values

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        property_values = {}
        for column in self._columns:
            if hasattr(column, property_name):
                property_values.update(getattr(column, property_name)(**kwargs))

        return property_values


class LimitClause(Clause):

    def __init__(self, number: int):
        self._number = number

    @classmethod
    def class_name(cls) -> str:
        return "limit"

    @classmethod
    def abbr(cls) -> str:
        return "L"

    def _get_property_list(self, property_name: str, **kwargs) -> list:
        return []

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        return {}


class SetOperator(Clause):
    _members = ["intersect", "union", "except"]

    def __init__(self, op: str, queries: tuple[Union['QueryInfo', 'SetOperator'], Union['QueryInfo', 'SetOperator']]):
        self.op = op
        self.queries = queries

    @classmethod
    def is_query(cls) -> bool:
        """ Returns if the class can represent a query. """
        return True

    @classmethod
    def class_name(cls) -> str:
        return "set_operator"

    @classmethod
    def abbr(cls) -> str:
        return "SO"

    @classmethod
    def members(cls):
        return cls._members

    def subqueries(self, shallow_search: bool = True) -> list['QueryInfo']:
        """
        Returns a list with the subqueries existing in both of the queries of the set operator.
        """
        subqs = []
        for query in self.queries:
            subqs.extend(query.subqueries())
        return subqs

    def setOperators(self, shallow_search: bool = False) -> list['SetOperator']:
        setOps = []
        for query in self.queries:
            setOps.extend(query.setOperators())
        return setOps

    def _get_property_list(self, property_name: str, **kwargs) -> list[str]:
        property_values = []
        for query in self.queries:
            property_values.extend(getattr(query, property_name)(**kwargs))
        return property_values

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        property_values = {}
        for query in self.queries:
            property_values.update(getattr(query, property_name)(**kwargs))
        return property_values

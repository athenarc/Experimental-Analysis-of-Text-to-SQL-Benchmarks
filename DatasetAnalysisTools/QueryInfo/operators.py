from abc import ABC
from typing import Union, Type, Any, Literal

from DatasetAnalysisTools.QueryInfo.variables import Value, Column, Table, QueryAsTable


class Operator(ABC):

    def __init__(self, op: str,
                 operands: Union[
                     tuple[Union[Value, Column, 'Operator', 'QueryInfo', 'SetOperator', 'Join', Table, QueryAsTable],
                     Union[Value, Column, 'Operator', 'QueryInfo', list, 'Join', Table, QueryAsTable, 'SetOperator']],
                     tuple[Union[Column, str]]]):
        self.op = op
        self.operands = operands

    @classmethod
    def is_query(cls) -> bool:
        """ Returns if the class can represent a query. """
        return False

    def _get_property_list(self, property_name: str, **kwargs) -> list[str, 'Operator']:
        property_values = []
        for operand in self.operands:
            if not ("shallow_search" in kwargs and kwargs["shallow_search"] and hasattr(operand, "is_query") and
                    operand.is_query()):
                if hasattr(operand, property_name):
                    property_values.extend(getattr(operand, property_name)(**kwargs))
        return property_values

    def tables(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        """
        Returns the tables of the schema existing in an operator or in any nested query in an operator
        """
        return self._get_property_list("tables", **{"shallow_search": shallow_search,
                                                    "unique": unique})

    def values(self, shallow_search: bool = False) -> list[Any]:
        """" Returns the values existing in an operator or in any nested query in an operator. """
        vals = []
        for operand in self.operands:
            operand_type = type(operand)
            if operand_type is Value:
                vals.append(operand.name)
            elif hasattr(operand, "values"):
                if not (shallow_search and hasattr(operand, "is_query") and operand.is_query()):
                    vals.extend(operand.values(shallow_search=shallow_search))
            elif operand_type is list:  # in case of in with list
                for subop in operand:
                    vals.append(subop.name)
        return vals

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """
        Returns the columns of the schema existing in an operator.
        """
        cols = []
        for operand in self.operands:
            operand_type = type(operand)
            if operand_type is Column:
                if return_format == "name":
                    cols.append(operand.name)
                elif return_format == "complete_name":
                    cols.append(operand.fullname)
                else:  # return_type == "raw"
                    cols.append(operand)
            elif hasattr(operand, "columns"):
                if not (shallow_search and hasattr(operand, "is_query") and operand.is_query()):
                    cols.extend(operand.columns(return_format=return_format, shallow_search=shallow_search,
                                                unique=unique))
        return cols

    def subqueries(self, shallow_search: bool = True) -> list['QueryInfo']:
        """ Returns a list of subqueries existing in the operator. """
        subqs = []
        for operand in self.operands:
            if hasattr(operand, "is_query") and operand.is_query():
                # If operand is a set operator
                if hasattr(operand, "queries"):
                    subqs.extend(list(operand.queries))
                else:  # If operand is a QueryInfo
                    subqs.append(operand)
            elif hasattr(operand, "subqueries"):
                subqs.extend(operand.subqueries())
        return subqs

    def setOperators(self, shallow_search: bool = False) -> list['SetOperator']:
        """ Returns a list of set operators existing in the operator. """
        setOps = []
        for operand in self.operands:
            try:
                if operand.class_name() == "set_operator":
                    setOps.append(operand)
                    continue
            except (AttributeError, TypeError):
                pass
            if hasattr(operand, "setOperators"):
                setOps.extend(operand.setOperators())
        return setOps

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

    def aliases(self, shallow_search: bool = False) -> dict[str, str]:
        """
        Returns a dictionary with the aliases and the corresponding schema element names. If the alias is not for
        a schema element the value is None.
        """
        als = {}
        for operand in self.operands:
            if not (shallow_search and hasattr(operand, "is_query") and operand.is_query()):
                if hasattr(operand, "aliases"):
                    als.update(getattr(operand, "aliases")(**{"shallow_search": shallow_search}))
        return als


class Join(Operator):
    _members = ['join', 'left join', 'inner join', 'cross join', 'left outer join']

    def __init__(self, op: str, operands: tuple[Union['Join', Table, QueryAsTable], Union['Join', Table, QueryAsTable]],
                 on_condition: 'NumericalOperator' = None):
        super().__init__(op=op, operands=operands)
        self._on_condition = on_condition

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "join"

    @classmethod
    def abbr(cls) -> str:
        return "J"

    def tables(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        """
        Returns the tables of the schema existing in the join's tables or in any nested query.
        """
        schema_tables = []
        for operand in self.operands:
            if type(operand) is Table:
                schema_tables.append(operand.name)
            else:
                if not (shallow_search and operand.is_query()):
                    schema_tables.extend(operand.tables(shallow_search=shallow_search,
                                                        unique=unique))
        return schema_tables

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """
        Returns the columns of the schema existing in the join's tables or in any nested query.
        """
        cols = []
        for operand in self.operands:
            if type(operand) is not Table:
                if not (shallow_search and operand.is_query()):
                    cols.extend(operand.columns(return_format=return_format, shallow_search=shallow_search,
                                                unique=unique))

        if self._on_condition is not None:
            cols.extend(self._on_condition.columns(return_format=return_format, shallow_search=shallow_search,
                                                   unique=unique))

        return cols

    def subqueries(self, shallow_search: bool = True) -> list['QueryInfo']:
        """
        Returns a list of subqueries existing in the join.
        """
        subqs = []
        for operand in self.operands:
            if type(operand) is not Table:
                subqs.extend(operand.subqueries())

        if self._on_condition is not None:
            subqs.extend(self._on_condition.subqueries())

        return subqs

    def joins(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, Operator]:
        return super().joins(shallow_search=shallow_search) + [self.op if return_format == "name" else self]

    def _get_property_list(self, property_name: str, **kwargs) -> list[str, Operator]:
        property_values = super()._get_property_list(property_name, **kwargs)
        if hasattr(self._on_condition, property_name):
            property_values.extend(getattr(self._on_condition, property_name)(**kwargs))
        return property_values


class Aggregate(Operator):
    _members = ['count', 'min', 'max', 'sum', 'avg']

    def __init__(self, op: str, operands: tuple[Union[Column, str, Operator]]):
        super().__init__(op, operands)

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "aggregate"

    @classmethod
    def abbr(cls) -> str:
        return "Ag"

    def aggregates(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        return super().aggregates(shallow_search=shallow_search) + [self.op if return_format == "name" else self]


class ComparisonOperator(Operator):
    _members = ['gt', 'gte', 'lt', 'lte', 'eq', 'neq', 'between', 'not_between']

    def __init__(self, op: str, operands: tuple[Union[Value, Column, Operator, 'QueryInfo', 'SetOperator'],
    Union[Value, Column, Operator, 'QueryInfo', 'SetOperator'],
    Union[Value, Column, Operator, 'QueryInfo', 'SetOperator']]):
        super().__init__(op, operands)

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "comparison_operator"

    @classmethod
    def abbr(cls) -> str:
        return "C"

    def comparison_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        return super().comparison_operators(shallow_search=shallow_search) + [self.op if return_format == "name" else self]


class LogicalOperator(Operator):
    _members = ['and', 'or', 'not']

    def __init__(self, op: str, operands: tuple[Operator, Operator]):
        super().__init__(op, operands)

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "logical_operator"

    @classmethod
    def abbr(cls) -> str:
        return "Lo"

    def logical_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        return super().logical_operators(shallow_search=shallow_search) + [self.op if return_format == "name" else self]


class LikeOperator(Operator):
    _members = ['like', 'not_like', 'ilike']

    def __init__(self, op: str, operands: tuple[Union[Column, 'QueryInfo', 'SetOperator'], str]):
        super().__init__(op, operands)

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "like_operator"

    @classmethod
    def abbr(cls) -> str:
        return "Li"

    def like_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        return super().like_operators(shallow_search=shallow_search) + [self.op if return_format == "name" else self]


class ArithmeticOperator(Operator):
    _members = ['sub', 'add', 'mul', 'div']

    def __init__(self, op: str, operands: tuple[Union[Column, 'QueryInfo', 'SetOperator', Value],
                 Union[Column, 'QueryInfo', 'SetOperator', Value]]):
        super().__init__(op, operands)

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "arithmetic_operator"

    @classmethod
    def abbr(cls) -> str:
        return "Ar"

    def arithmetic_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        return super().arithmetic_operators(shallow_search=shallow_search) + [self.op if return_format == "name" else self]


class MembershipOperator(Operator):
    _members = ['in', 'nin']

    def __init__(self, op: str, operands: tuple[Union[Column, 'QueryInfo', 'SetOperator'],
                 Union[list[Value], 'QueryInfo', 'SetOperator']]):
        super().__init__(op, operands)

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "membership_operator"

    @classmethod
    def abbr(cls) -> str:
        return "Me"

    def membership_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        return super().membership_operators(shallow_search=shallow_search) + [self.op if return_format == "name" else self]


class NullOperator(Operator):
    _members = ["is null", "is not null"]

    def __init__(self, op: str, operands: tuple[Union[Column, 'QueryInfo', 'SetOperator']]):
        super().__init__(op, operands)

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "null_operator"

    @classmethod
    def abbr(cls) -> str:
        return "Nu"

    def null_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> list[str]:
        return super().null_operators(shallow_search=shallow_search) + [self.op if return_format == "name" else self]


class Distinct(Operator):
    _members = ["distinct"]

    def __init__(self, op: str, operands: tuple[Column]):
        for operand in operands:
            if type(operand) is not Column:
                raise TypeError("Distinct clause can contain only columns!")
        super().__init__(op=op, operands=operands)

    @classmethod
    def members(cls):
        return cls._members

    @classmethod
    def class_name(cls) -> str:
        return "distinct"

    @classmethod
    def abbr(cls) -> str:
        return "D"

    def tables(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        return []

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        if return_format == "name":
            return [operand.name for operand in self.operands]
        elif return_format == "complete_name":
            return [operand.fullname for operand in self.operands]
        else:  # return_format == "raw
            return [operand for operand in self.operands]

    def subqueries(self, shallow_search: bool = True) -> list['QueryInfo']:
        return []


def select_operator_type(operator_name: str) -> Union[Type[Operator], None]:
    """
    Returns the class of an operator based on its name.
    Args:
        operator_name (str): The name of an operator (e.g., count)
    """
    for operator_type in [Join, Aggregate, ComparisonOperator, LogicalOperator, LikeOperator, ArithmeticOperator,
                          MembershipOperator, NullOperator, Distinct]:
        if operator_name in operator_type.members():
            return operator_type

    return None


def select_condition_type(operator_name: str) -> \
        Union[Type[Union[ComparisonOperator, LogicalOperator, LikeOperator, ArithmeticOperator, MembershipOperator,
        NullOperator]],
        None]:
    """
    Returns the class of a condition based in its name.
    Condition is an operator excluding aggregates
    Args:
        operator_name (str): The name of an operator (e.g., count)
    """
    for operator_type in [Join, ComparisonOperator, LogicalOperator, LikeOperator, ArithmeticOperator,
                          MembershipOperator, NullOperator]:
        if operator_name in operator_type.members():
            return operator_type

    return None

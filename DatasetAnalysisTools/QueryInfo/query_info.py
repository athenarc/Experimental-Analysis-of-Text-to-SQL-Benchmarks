from typing import Union, Literal, Any

from DatasetAnalysisTools.DatabaseInfo.database_info import DatabaseInfo
from DatasetAnalysisTools.QueryInfo.clause import SetOperator, SelectClause, FromClause, WhereClause, GroupByClause, \
    HavingClause, OrderByClause, LimitClause
from DatasetAnalysisTools.QueryInfo.operators import Join, Aggregate, LogicalOperator, LikeOperator, ArithmeticOperator, \
    ComparisonOperator, MembershipOperator, NullOperator, Operator
from DatasetAnalysisTools.QueryInfo.variables import Column, Table

CLAUSES = [SelectClause, FromClause, WhereClause, GroupByClause, HavingClause, OrderByClause, LimitClause]

OPERATOR_TYPES = [Join, Aggregate, ComparisonOperator, LogicalOperator, LikeOperator, ArithmeticOperator,
                  MembershipOperator, NullOperator]

VARIABLE_TYPES = ["columns", "tables", "values"]


class QueryInfo:

    def __init__(self, content: Union[SetOperator, tuple[SelectClause, FromClause, Union[WhereClause, None],
                 Union[GroupByClause, None], Union[HavingClause, None],
                 Union[OrderByClause, None], Union[LimitClause, None]]], database_info: DatabaseInfo = None):
        """
        Initializes the QueryInfo structure.
        Args:
            content: The clauses of the query
            database_info (DatabaseInfo): The information about the database upon which the query is made. It is not
                required to be provided, but without it some information will be incomplete.
                Example: In the query 'select name from author join book on author.id = book.author_id where
                author.age < 18' the table of the column 'name' will be None due to lack of information.
        """

        # Initialize the clauses
        if type(content) == SetOperator:
            self.selectClause = None
            self.fromClause = None
            self.whereClause = None
            self.groupByClause = None
            self.havingClause = None
            self.orderByClause = None
            self.limitClause = None
            self.setOperator = content
        else:
            self.selectClause = content[0]
            self.fromClause = content[1]
            self.whereClause = content[2]
            self.groupByClause = content[3]
            self.havingClause = content[4]
            self.orderByClause = content[5]
            self.limitClause = content[6]
            self.setOperator = None

            # Add implicit tables in columns
            self._add_implicit_tables(database_info)

    @classmethod
    def is_query(cls) -> bool:
        """ Returns if the class can represent a query. """
        return True

    def _add_implicit_tables(self, database_info: DatabaseInfo = None) -> None:
        """
        Adds the table name in the columns that are not explicitly mentioned.
        Without the schema implicit tables can be added only in case of 1 table in the 'from' clause!.

        Args:
            database_info (DatabaseInfo): The information of the query's database.
        """
        # If the information about the database are not provided and the query has only 1 table
        if database_info is None:
            query_tables = self.fromClause._tables
            # If there is only one table in the query
            if len(query_tables) == 1 and not isinstance(query_tables[0], Join):
                implicit_table_name = query_tables[0].name \
                    if isinstance(query_tables[0], Table) else query_tables[0].alias
                # Add the table name in the columns that is not explicitly stated
                for column in self.columns(return_format="raw", shallow_search=True):
                    # If the table name
                    if column.table_name is None:
                        column.table_name = implicit_table_name

    def get_columns_equivalences(self, shallow_search: bool = True) -> dict[
                                 Union[str, Column], list[Union[str, Column]]]:
        """
        Returns the columns in the query that can be used interchangeable due to equality condition. The columns are
        stored in they dictionary with their fullname, with the actual table name (if possible)

        Args:
            shallow_search (bool): If True the columns of the subqueries will not be
                considered.

        !! Currently the shallow_search = False is not supported
        """

        aliases = self.aliases(shallow_search=shallow_search)
        tables = self.tables(shallow_search=shallow_search, unique=True)

        # Get the equality conditions of the query
        comp_ops = self.comparison_operators(shallow_search=shallow_search, return_format="raw")
        eq_ops = [op for op in comp_ops if op.op]

        # Create an equivalences dictionary based on the equality conditions
        equivalences = {}
        for eq_op in eq_ops:
            operands = eq_op.operands
            # If the operands are both columns
            if isinstance(operands[0], Column) and isinstance(operands[1], Column):

                columns = self._replace_table_aliases(operands, aliases, tables)

                if columns[0] in equivalences:
                    equivalences[columns[0]].append(columns[1])
                else:
                    equivalences[columns[0]] = [columns[1]]

                if columns[1] in equivalences:
                    equivalences[columns[1]].append(columns[0])
                else:
                    equivalences[columns[1]] = [columns[0]]

        return equivalences

    @staticmethod
    def _replace_table_aliases(columns: list[Column], aliases: dict, tables: list[str]) -> list[str]:

        columns_without_aliases = []
        for column in columns:
            # If the table in the column is not in the schema tables and exists in the aliases of the query
            if column.table_name is not None and column.table_name not in tables and column.table_name in aliases:
                columns_without_aliases.append(f"{aliases[column.table_name]}.{column.name}")
            else:
                columns_without_aliases.append(column.fullname)
        return columns_without_aliases

    def columns_without_table_aliases(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        """
        Replace the table name in the columns if it is an alias to a schema table and returns the
        "<table_name>.<column_name>" for each column in the query, if the information about the table are available else
        only the name is returned.

        !!! In case of a query as table the alias remains.
        !!! Current version does not support alias visibility to other queries (inner or outer)

        Args:
            shallow_search (bool): If True the columns of the subqueries are not included in the returned list.
            unique (bool): If True only the unique values are kept.
        """
        aliases = self.aliases(shallow_search=True)
        tables = self.tables(shallow_search=True, unique=True)

        cols = self._replace_table_aliases(self.columns(return_format="raw", shallow_search=True, unique=unique),
                                           aliases, tables)

        if not shallow_search:
            for subquery in self.subqueries():
                cols.extend(subquery.columns_without_table_aliases(shallow_search=False, unique=unique))

        if unique:
            cols = list(set(cols))

        return cols

    def _get_property_list(self, property_name: str, clauses: list[str] = None, **kwargs) -> list[Any]:
        """
        Returns a list with the values of the property in the requested clauses.
        Args:
            property_name (str): The name of the property that will be accessed.
            clauses (list(str)): A list of clauses that will access for the property values. If None,
                all clauses are considered.
            shallow_search (bool): If True the structural components of the subqueries will not be
                considered.
        """
        if clauses is None:
            clauses = [self.selectClause, self.fromClause, self.whereClause, self.groupByClause, self.havingClause,
                       self.orderByClause, self.limitClause, self.setOperator]

        property_values = []
        for clause in clauses:
            if clause is not None and hasattr(clause, property_name):
                property_values.extend(getattr(clause, property_name)(**kwargs))
        return property_values

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        property_values = {}
        for clause in [self.selectClause, self.fromClause, self.whereClause, self.groupByClause, self.havingClause,
                       self.orderByClause, self.limitClause, self.setOperator]:
            if clause is not None and hasattr(clause, property_name):
                property_values.update(getattr(clause, property_name)(**kwargs))
        return property_values

    def subqueries(self) -> list['QueryInfo']:
        """ Returns a list with the subqueries existing in the query. """
        return self._get_property_list("subqueries")

    def aliases(self, shallow_search: bool = False) -> dict[str, str]:
        """
        Returns a dictionary with the aliases and the corresponding schema element names. If the alias is not for
        a schema element the value is None.
        """
        return self._get_property_dict("aliases", **{"shallow_search": shallow_search})

    def joins(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, Operator]:
        """
        Returns the joins existing in the query or in any subquery.

        Args:
            shallow_search (bool): If True the joins of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("joins", **{"shallow_search": shallow_search,
                                                   "return_format": return_format})

    def aggregates(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, Operator]:
        """
        Returns the aggregates existing in the query.

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
            list[str, Operator]:
        """
        Returns the comparison operators existing in the query.

        Args:
            shallow_search (bool): If True the comparison operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("comparison_operators", **{"shallow_search": shallow_search,
                                                                  "return_format": return_format})

    def logical_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, Operator]:
        """
        Returns the logical operators existing in the query.

        Args:
            shallow_search (bool): If True the logical operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("logical_operators", **{"shallow_search": shallow_search,
                                                               "return_format": return_format})

    def like_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, Operator]:
        """
        Returns the like operators existing in the query.

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
            list[str, Operator]:
        """
        Returns the arithmetic operators existing in the query.

        Args:
            shallow_search (bool): If True the arithmetic operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list(property_name="arithmetic_operators", **{"shallow_search": shallow_search,
                                                                                "return_format": return_format})

    def membership_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, Operator]:
        """
        Returns the membership operators existing in the query.

        Args:
            shallow_search (bool): If True the membership of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("membership_operators", **{"shallow_search": shallow_search,
                                                                  "return_format": return_format})

    def null_operators(self, shallow_search: bool = False, return_format: Literal["name", "raw"] = "name") -> \
            list[str, Operator]:
        """
        Returns the null operators existing in the query.

        Args:
            shallow_search (bool): If True the null operators of the subqueries will not be
                considered.
            return_format: (Literal["name", "raw"] = "name"): The format of the returned operators.
                - name: The name of each operator is returned.
                - raw: The class of each operator is returned.
        """
        return self._get_property_list("null_operators", **{"shallow_search": shallow_search,
                                                            "return_format": return_format})

    def tables(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        """
        Returns the names of the schema tables existing in the query.

        Args:
            shallow_search (bool): If True the tables of the subqueries will not be
                considered.
            unique (bool): If True only the unique values are kept.
        """

        t = self._get_property_list("tables", **{"shallow_search": shallow_search})

        if unique:
            t = list(set(t))

        return t

    def columns(self, return_format: Literal["name", "complete_name", "raw"] = "complete_name",
                shallow_search: bool = False, unique: bool = False) -> list[Union[str, Column]]:
        """
        Returns the columns existing in the query.

        Args:
            return_format (Literal["name", "complete_name", "raw"]): The format of the returned columns.
                - name: Only the names of the columns are returned
                - complete_name: The name of the table followed by the column name is returned for each column
                - raw: A class of type Column is returned for each column
            shallow_search (bool): If True the columns of the subqueries will not be
                considered.
            unique (bool): If True only the unique values are kept.
        """

        cols = self._get_property_list("columns",
                                       **{"shallow_search": shallow_search, "return_format": return_format,
                                          "unique": unique})

        if unique:
            cols = list(set(cols))

        return cols

    def values(self, shallow_search: bool = False) -> list[Any]:
        """
        Returns the values (e.g., literals, numbers) existing in the query.

        Args:
            shallow_search (bool): If True the values of the subqueries will not be
                considered.
        """
        return list(self._get_property_list("values", **{"shallow_search": shallow_search}))

    def functions(self, shallow_search: bool = False):
        """
        Returns the functions existing in the query.
        """
        pass

    def setOperators(self, shallow_search: bool = False) -> list['SetOperator']:
        return self._get_property_list("setOperators", **{"shallow_search": shallow_search})

    def depth(self) -> int:
        """ Returns the depth of the query """
        return max([subquery.depth() + 1 for subquery in self.subqueries()] + [0])

    def structural_components(self, shallow_search: bool = False, with_names: bool = False) -> list[str]:
        """
        Returns a list with the structural components existing in the query.
        Args:
            shallow_search (bool): If True the structural components of the subqueries will not be considered.
            with_names (bool): If true the structural components are returned with their names, else with their
                abbreviations.
        """
        components = []

        if self.setOperator is not None:
            components.append(self.setOperator.class_name() if with_names else self.setOperator.abbr())
            for query in self.setOperator.queries:
                components.extend(query.structural_components(shallow_search=True, with_names=with_names))
        else:
            for clause in [self.selectClause, self.fromClause, self.whereClause, self.groupByClause, self.havingClause,
                           self.orderByClause, self.limitClause]:
                if clause is not None:
                    components.append(clause.class_name() if with_names else clause.abbr())

        components.extend([SetOperator.class_name() if with_names else SetOperator.abbr()
                           for _ in range(len(self.setOperators()))])

        nestings_num = len(self.subqueries()) - len(self.setOperators())
        components.extend(["nesting" if with_names else "N" for _ in range(nestings_num)])

        if not shallow_search:
            for subquery in self.subqueries():
                components.extend(subquery.structural_components(shallow_search=False, with_names=with_names))

        return components

    def operatorTypes(self, shallow_search: bool = False, with_names: bool = False, per_clause: bool = False) -> \
            Union[list[str], dict[str, list[str]]]:
        """
        Returns all the operator types existing in the query.
        If the per_clause parameter is True the operators are returned per clause.
        Args:
            shallow_search (bool): If True the structural components of the subqueries will not be
                considered.
            with_names (bool): If true the structural components are returned with their names, else with their
                abbreviations.
            per_clause (bool): If True the method returns the operator types per clause.

        """
        operatorTypes = {} if per_clause else []

        for clause in [self.selectClause, self.fromClause, self.whereClause, self.groupByClause, self.havingClause,
                       self.orderByClause, self.limitClause, self.setOperator]:
            if clause is not None:
                if per_clause:
                    operatorTypes[clause.class_name()] = clause.operatorTypes(shallow_search=shallow_search,
                                                                              with_names=with_names)
                else:
                    operatorTypes.extend(clause.operatorTypes(shallow_search=shallow_search, with_names=with_names))

        return operatorTypes

    def operators(self, shallow_search: bool = False, per_clause: bool = False) -> \
            Union[list[str], dict[str, list[str]]]:
        """Returns all the operators existing in the query.
        """
        operators = {} if per_clause else []

        for clause in [self.selectClause, self.fromClause, self.whereClause, self.groupByClause, self.havingClause,
                       self.orderByClause, self.limitClause, self.setOperator]:
            if clause is not None:
                if per_clause:
                    operators[clause.class_name()] = clause.operators(shallow_search=shallow_search)
                else:
                    operators.extend(clause.operators(shallow_search=shallow_search))
        return operators

    @staticmethod
    def _get_category(category_components: list[str], query_components: list[str],
                      return_format: Literal["binary", "counter", "name"]):
        """
        Returns the category of the query based on the category components and the components found in the query.
        Args:
            category_components (list[str]): The components that comprise the category.
            query_components (list[str]): The components found in the query.
            return_format (str): The format in which the category will be returned.
                - 'binary': Each category's component will be 0 or 1, depending on the existence or not in the query.
                - 'counter': Each category's component will represented by the number of appearances in the query.
                - 'name': The category will contain the name of a component if the component appears in the query.
        """
        category = []

        for component in category_components:
            if return_format == "counter":
                category.append(str(query_components.count(component)))
            else:
                component_exists = component in query_components

                if return_format == "binary":
                    category.append(str(1 if component_exists else 0))
                else:  # If return_format == "name"
                    if component_exists:
                        category.append(component)

        if return_format == "binary":
            return "".join(category)
        elif return_format == "name":
            return "-".join(category)
        else:  # if return_format == "counter"
            # sep is added for counter values > 9
            return ".".join(category)

    def get_structural_category(self, return_format: Literal["binary", "counter", "name"],
                                shallow_search: bool = False) -> str:
        """
        Returns the structural category of the query. The structural category is defined as the set of the structural
        components existing in the query.

        Structural components : [select(S), from(F), where(W), group by(G), order by(O), having(H), limit(L),
        set operator(SO), nesting(N)]

        Args:
            return_format (str): The format in which the category will be returned.
                - 'binary': Each category's component will be 0 or 1, depending on the existence or not of a structural
                    component.
                - 'counter': Each category's component with be the number of appearances of the corresponding structural
                    component.
                - 'name': The category will contain the abbreviations of a structural component if the component
                    appears in the query.
            shallow_search (bool): If True the structural components of the subqueries will not be
                considered.

        Returns (str): The structural category of the query.

        """
        query_structural_components = self.structural_components(shallow_search=shallow_search)

        structural_components_abbrs = [SelectClause.abbr(), FromClause.abbr(), WhereClause.abbr(), GroupByClause.abbr(),
                                       HavingClause.abbr(), OrderByClause.abbr(), LimitClause.abbr(),
                                       SetOperator.abbr(), "N"]

        return self._get_category(category_components=structural_components_abbrs,
                                  query_components=query_structural_components,
                                  return_format=return_format)

    def get_operatorTypes_category(self, return_format: Literal["binary", "counter", "name"],
                                   shallow_search: bool = False) -> str:
        """
        Returns the operatorTypes category of the query. The operatorTypes category is defined as the set of the
        operator types existing in the query.

        Operator types: [joins(J), aggregates(Ag), comparison_operators(C), logical_operators(Lo), like_operators(Li),
                        arithmetic_operators(Ar), membership_operators(Me), null_operators(Nu)]

        Args:
            return_format: The format in which the category will be returned.

                - 'binary': Each category's component will be 0 or 1, depending on the existence or not of an operator type.
                - 'counter': Each category's component with be the number of appearances of the corresponding operator type.
                - 'name': The category will contain the name of an operator type if it appears in the query.
            shallow_search (bool): If True the structural components of the subqueries will not be
                                                   considered.

        Returns (str): The operatorTypes category of the query.
        """
        query_operatorTypes = self.operatorTypes(shallow_search=shallow_search)

        operatorTypes_abbrs = ["J", "Ag", "C", "Lo", "Li", "Ar", "Me", "Nu"]

        return self._get_category(category_components=operatorTypes_abbrs,
                                  query_components=query_operatorTypes,
                                  return_format=return_format)


def get_subqueries(queries: list[QueryInfo]) -> list[QueryInfo]:
    """
    Returns a list with the subqueries of all the given queries.
    """
    subs = []
    for query in queries:
        subs.extend(query.subqueries())
    return subs


def get_subqueries_per_depth(query_info: QueryInfo) -> dict[int, list]:
    """
    Returns a dictionary with the depth and the subqueries that correspond the depth in the given query.
    Args:
        query_info (QueryInfo): The given query_info class
    """
    depth = 0
    subqueries_per_depth = [[query_info]]

    while len(subqueries_per_depth[-1]) > 0:
        subqueries_per_depth.append(get_subqueries(subqueries_per_depth[-1]))

    # Remove the last entry which has 0 subqueries
    subqueries_per_depth = subqueries_per_depth[:-1]

    return {depth: depth_subqueries for depth, depth_subqueries in enumerate(subqueries_per_depth)}

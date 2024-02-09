from typing import Union

from mo_parsing import ParseException, ParseSyntaxException, RecursiveGrammarException
from mo_sql_parsing import parse_mysql

from DatasetAnalysisTools.QueryInfo.clause import SelectClause, SetOperator, FromClause, WhereClause, GroupByClause, \
    HavingClause, OrderByClause, LimitClause
from DatasetAnalysisTools.QueryInfo.operators import Operator, select_operator_type, Join, select_condition_type, \
    ComparisonOperator
from DatasetAnalysisTools.QueryInfo.query_extractor.query_extractor_abc import QueryExtractor
from DatasetAnalysisTools.QueryInfo.query_info import QueryInfo
from DatasetAnalysisTools.QueryInfo.variables import Column, Value, Table, QueryAsTable, OperatorAsColumn
from DatasetAnalysisTools.DatabaseInfo.database_info import DatabaseInfo
from exceptions import SQLParserException


class MoQueryExtractor(QueryExtractor):
    _parser_clauses = ['select_distinct', 'select', 'from', 'where', 'groupby', 'having', 'orderby', 'limit']

    def extract(self, query: str, db_info: DatabaseInfo = None) -> QueryInfo:
        """
        Gets a query in a string format and instantiates a QueryInfo class.
        Args:
            query (str): An sql query in a string format
            db_info (DatabaseInfo): The information (e.g, the table names) of the database upon which the query is made.
        """

        # Parse the sql query with the use of the mozilla sql parser
        try:
            parsed_query = parse_mysql(query)
        except (ParseException, ParseSyntaxException, RecursiveGrammarException) as e:
            raise SQLParserException(e)

        try:
            query_info = self._extract_query_info(parsed_query)
        except SyntaxError as e:
            raise SyntaxError(f"There is a syntax error in the query {query}")

        return query_info

    def _extract_query_info(self, parsed_query: dict) -> QueryInfo:
        """
        Instantiates a QueryInfo with the information of the parsed query.
        Args:
            parsed_query (dict): A sql query parsed with the mozilla sql parser
        """
        query_content = []

        # If the query is a set operator
        if any([key in SetOperator.members() for key in parsed_query.keys()]):
            set_operator_type = list(parsed_query.keys())[0]
            query_content = self._extract_set_operator(set_operator_type=set_operator_type,
                                                       set_operator_content=parsed_query[set_operator_type])
        else:  # Extract all clauses
            for clause in self._parser_clauses:
                if clause in parsed_query.keys():
                    # Extract the contents of the clause
                    query_content.append(getattr(self, f"_extract_{clause}_info")(parsed_query[clause]))
                else:
                    if clause not in ["select", "select_distinct", "from"]:
                        query_content.append(None)

        return QueryInfo(content=query_content)

    def _extract_set_operator(self, set_operator_type: str, set_operator_content: list[dict]) -> SetOperator:
        """
        Instantiates a SetOperator with the provided operator type (e.g., intersect) and the content
        Args:
            set_operator_type (str): The type of the set operator (e.g., intersect)
            set_operator_content (list[dict]): The parsed content of the set operator
        """

        return SetOperator(op=set_operator_type, queries=(self._extract_query_info(set_operator_content[0]),
                                                          self._extract_query_info(set_operator_content[1])))

    def _extract_select_info(self, select_content: Union[list, str, dict], distinct=False) -> SelectClause:
        """
        Instantiates a SelectClause with the provided content.
        Args:
            select_content (Union[list, str]): The parsed content of a select clause
        """
        attributes = []
        select_type = type(select_content)

        # If the select contains multiple attributes
        if select_type == list:
            for attribute_raw in select_content:
                attributes.append(self._extract_column_info(attribute_raw))
        # If the select clause contains only a '*'
        elif select_type == str:
            attributes.append('*')
        else:  # If the select contains only one attribute
            attributes.append(self._extract_column_info(select_content))

        return SelectClause(attributes, distinct=distinct)

    def _extract_select_distinct_info(self, select_content: list) -> SelectClause:
        """
        Instantiates a SelectClause with the provided content.
        (The distinct keyword of the select is set ot true as the parsed keyword is 'select_distinct')
        Args:
            select_content: (Union[list, str]): The parsed content of a select clause
        """
        return self._extract_select_info(select_content=select_content, distinct=True)

    def _extract_from_info(self, from_content_raw: Union[str, list, dict]) -> FromClause:
        """
        Instantiates a FromClause with the provided content.
        Args:
            from_content_raw (Union[list, str]): The parsed content of a from clause
        """

        from_content_type = type(from_content_raw)
        from_content = []

        # If the from clause has only one table
        if from_content_type is str:
            from_content.append(Table(name=from_content_raw))
        # If the from clause has multiple tables
        elif from_content_type is list:
            from_content.extend(self._extract_joins_info(from_content_raw))
        else:  # If the from clause is a dict (e.g., table with alias)
            from_content.append(self._extract_table_info(from_content_raw))

        return FromClause(tables=from_content)

    def _extract_where_info(self, where_content_raw: dict) -> WhereClause:
        """
        Instantiates a WhereClause with the provided content.
        Args:
            where_content_raw (dict): The parsed content of the where clause
        """

        # If the length of the given dict is greater than one there is a syntax error in the query
        if len(list(where_content_raw.keys())) > 1:
            raise SyntaxError()

        condition = self._extract_operator(where_content_raw)

        return WhereClause(condition=condition)

    def _extract_groupby_info(self, groupby_content_raw: Union[list, dict, str]) -> GroupByClause:
        """
        Instantiates a GroupByClause with the provided content.
        Args:
            groupby_content_raw (Union[list, dict, str]): The parsed content of the groupby clause
        """
        content_type = type(groupby_content_raw)
        columns = []

        # If group by has multiple columns
        if content_type is list:
            for column_raw in groupby_content_raw:
                columns.append(self._extract_column_info(column_raw))
        else:  # If group by has only one column
            columns = [self._extract_column_info(groupby_content_raw)]

        return GroupByClause(columns)

    def _extract_having_info(self, having_content_raw: dict) -> HavingClause:
        """
        Instantiates a HavingClause with the provided content.
        Args:
            having_content_raw (dict): The parsed content of the having clause
        """
        return HavingClause(condition=self._extract_operator(having_content_raw))

    def _extract_orderby_info(self, orderby_content_raw: Union[list, dict]) -> OrderByClause:
        """
        Instantiates a OrderByClause with the provided content.
        Args:
            orderby_content_raw (Union[list, dict]): The parsed content of the orderby clause
        """

        content_type = type(orderby_content_raw)
        columns = []
        orders = []

        # If orderby has multiple columns
        if content_type is list:
            for column_info_raw in orderby_content_raw:
                columns.append(self._extract_column_info(column_info_raw))
                orders.append("asc" if "sort" not in column_info_raw.keys() else column_info_raw["sort"].lower())
        else:  # If order by has only one column
            columns.append(self._extract_column_info(orderby_content_raw))
            orders.append("asc" if "sort" not in orderby_content_raw.keys() else orderby_content_raw["sort"].lower())

        return OrderByClause(columns=columns, orders=orders)

    def _extract_limit_info(self, limit_info_raw: int) -> LimitClause:
        """
        Instantiates a LimitClause with the provided content.
        Args:
            limit_info_raw (int): The parsed content of the limit clause
        """
        return LimitClause(number=limit_info_raw)

    def _extract_joins_info(self, join_content_raw: list) -> list[Union[Join, Table, QueryAsTable]]:
        """
        Instantiates a Join with the provided content.
        Args:
            join_content_raw (dict): The parsed content of a join
        """
        pending_table = None
        for table in join_content_raw:

            # If the table has a join keyword
            if type(table) is not str and any(key in Join.members() for key in table.keys()):
                table_keys = list(table.keys())

                join_type = table_keys[0]
                on_condition = None

                # If the join has an on condition
                if len(table_keys) == 2:
                    op = list(table["on"].keys())[0]
                    on_condition = ComparisonOperator(op=op,
                                                      operands=tuple(self._extract_attribute_info(table["on"][op])))

                join = Join(op=join_type, operands=(pending_table, self._extract_table_info(table[join_type])),
                            on_condition=on_condition)
                pending_table = join
            else:

                if pending_table is not None:
                    join = Join(op="join", operands=(pending_table, self._extract_table_info(table)))
                    pending_table = join
                else:
                    pending_table = self._extract_table_info(table)

        return [pending_table]

    def _extract_table_info(self, table_content) -> Union[Table, QueryAsTable]:
        """
        Instantiates a Table or a QueryAsTable based on the provided content.
        Args:
            table_content: The parsed content of a table
        """
        table_content_type = type(table_content)

        # If the content is just a table name
        if table_content_type is str:
            return Table(name=table_content)
        elif table_content_type is dict:
            # If is a table with an alias
            if 'name' in table_content.keys():
                # If the table exists in the schema (table contains only a name)
                if type(table_content["value"]) is str:
                    return Table(name=table_content["value"], alias=table_content["name"])
                else:
                    return QueryAsTable(query=self._extract_query_info(table_content["value"]),
                                        alias=table_content["name"])
            else:  # If the table does not have an alias but is a nested query
                return QueryAsTable(query=self._extract_query_info(table_content))
        raise SyntaxError()

    def _extract_column_info(self, column_info_raw: Union[dict, str]) -> Union[Column, OperatorAsColumn]:
        """
        Extracts the information of a column. A column could be a schema column or the result of an operation
        Args:
            column_info_raw (dict): The extracted content of the column
        """

        def get_table_from_column_name(column_name: str) -> (Union[str, None], str):
            # If the column is referenced with its table name or with a table alias
            if "." in column_name:
                return column_name.split(".")
            else:
                return None, column_name

        # If the column content contain only the name of the column
        if type(column_info_raw) is str:
            table_name, column_name = get_table_from_column_name(column_info_raw)
            return Column(name=column_name, table_name=table_name)
        else:
            # Initialize the alias of a column
            alias = None
            if "name" in column_info_raw:
                alias = column_info_raw["name"]

            value_type = column_info_raw["value"]
            # If the value is a simple column
            if type(value_type) is str:
                table_name, column_name = get_table_from_column_name(column_info_raw["value"])
                return Column(name=column_name, table_name=table_name, alias=alias)
            else:  # If the columns is the result of an operation
                return OperatorAsColumn(operator=self._extract_operator(column_info_raw["value"]), alias=alias)

    def _extract_operator(self, operator_content_raw: dict) -> Operator:
        # Find the type of the operator
        op = list(operator_content_raw.keys())[0]
        opClass = select_operator_type(op)

        operands = self._extract_attribute_info(operator_content_raw[op])
        # If the operands are more than 2 and the operator cannot take more than 2 operands create an operator class
        # for every pair right to left.
        # E.g., if operator_content_raw = {'add': ['LastName', {'literal': ', '}, 'FirstName']} the returned operator
        # will be ArithmeticOperator('LastName', ArithmeticOperator({'literal': ', '}, 'FirstName'))
        if "between" not in op and op != "distinct" and "in" not in op:
            while len(operands) > 2:
                operands = operands[:-2] + [opClass(op=op, operands=tuple(operands[-2:]))]

        return opClass(op=op, operands=tuple(operands))

    def _extract_attribute_info(self, attribute_raw: Union[str, int, float, list, dict]) -> \
            list[Union[Operator, Column, QueryInfo, Value, SetOperator]]:
        """
        Extracts the information of the provided attribute's content into one of the valid attribute instances.
        An attribute can be a Column, an Operator, a QueryInfo, a SetOperator or a Value.
        Args:
            attribute_raw (Union[str, int, float, list, dict]): The parsed content of an attribute.
        """
        attribute_type = type(attribute_raw)
        attributes = []

        # If the attribute is a column
        if attribute_type is str:
            if attribute_raw == "*":
                attributes.append("*")
            else:
                attributes.append(self._extract_column_info(attribute_raw))

        # If attribute is a value of type value
        elif attribute_type in [int, float, str]:
            attributes.append(Value(name=attribute_raw))
        # If the attribute is a list of attributes
        elif attribute_type is list:
            for attr_raw in attribute_raw:
                attributes.extend(self._extract_attribute_info(attribute_raw=attr_raw))
        else:
            keys = list(attribute_raw.keys())
            # If the attribute is a subquery
            if "select" in keys or "select_distinct" in keys:
                attributes.append(self._extract_query_info(parsed_query=attribute_raw))
            else:
                if len(keys) > 1:
                    raise Exception()

                key = keys[0]

                if key == "literal":
                    literal_type = type(attribute_raw["literal"])
                    if literal_type in [str, float, int]:
                        attributes.append(Value(attribute_raw["literal"]))
                    elif literal_type is list:
                        for literal in attribute_raw["literal"]:
                            attributes.append(Value(literal))
                    else:
                        raise Exception()
                elif key == "value":
                    attributes.append(self._extract_column_info(attribute_raw))
                elif key in ["intersect", "union", "except"]:
                    attributes.append(self._extract_set_operator(set_operator_type=key,
                                                                 set_operator_content=attribute_raw[key]))
                else:  # If the attribute is an operator
                    attributes.append(self._extract_operator(attribute_raw))

        return attributes

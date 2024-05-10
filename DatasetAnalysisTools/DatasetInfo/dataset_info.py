import json
from typing import Union, Literal
from loguru import logger
import pandas as pd
from matplotlib import cbook
from tqdm import tqdm
from pathlib import Path

from DatasetAnalysisTools.QueryInfo.clause import SelectClause, WhereClause, GroupByClause, HavingClause, OrderByClause, \
    LimitClause, FromClause
from DatasetAnalysisTools.QueryInfo.operators import Join, MembershipOperator, Aggregate, ComparisonOperator, \
    LogicalOperator, LikeOperator, ArithmeticOperator, NullOperator
from DatasetAnalysisTools.QueryInfo.query_extractor.mo_sql_parser_extractor import MoQueryExtractor
from DatasetAnalysisTools.QueryInfo.query_info import QueryInfo, get_subqueries_per_depth
from DatasetAnalysisTools.DatabaseInfo.database_info import DatabaseInfo
from DatasetAnalysisTools.QuestionInfo.question_info import QuestionInfo, text_complexity_measures
from exceptions import SQLParserException


CLAUSES = [SelectClause, FromClause, WhereClause, GroupByClause, HavingClause, OrderByClause, LimitClause]

OPERATOR_TYPES = [Join, Aggregate, ComparisonOperator, LogicalOperator, LikeOperator, ArithmeticOperator,
                  MembershipOperator, NullOperator]

VARIABLE_TYPES = ["columns", "tables", "values"]

PARSING_ERRORS_DIR = "storage/error_logs"

DATASET_INFO_SAVE_DIR = "storage/datasets_info"


class DatasetInfo:
    """Used to store the information of a dataset and provide methods for statistics"""

    def __init__(self, dataset_name: str, dataset: list[dict], store_sql_queries_info: bool = False,
                 save: bool = False, ignore_info: list[Literal["databases", "questions", "sql_queries"]] = None):
        """
        Args:
            dataset_name (str): The name of the given dataset.
            dataset (list[dict]): A list with all the datapoints of the dataset. The dictionary of each datapoint must contain the following keys:

             - sql_query (str): The sql query
             - question (str): The natural language question corresponding to the sql query
             - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
                           upon which the queries are made. This field is optional
            store_sql_queries_info (bool): If true in the structure with the information of the sql queries will be stored the QueryInfo class of every sql query.
            save (bool): If True the initialized structures with the information of the dataset are stored in the <DATASET_INFO_SAVE_DIR> directory.
            ignore (list): A list with the dataset parts (questions, sql_queries, databases), whose information will be not calculated
        """
        self.ignore_info = ignore_info if ignore_info is not None else []

        self.name = dataset_name

        self.operators = []
        for operator_type in OPERATOR_TYPES:
            self.operators.extend(operator_type.members())

        self.variable_types = VARIABLE_TYPES

        # Initialize the columns of the dataframe that will store the information of all the datapoints
        self._queries_info_dict = self._initialize_queries_info_dict(store_sql_queries_info)

        # Initialize the columns of the dataframe that will store the information of all the databases
        self._dbs_info_dict = self._initialize_dbs_info_dict()

        # Initialize the columns of the dataframe that will store the information of all the natural language questions
        self._questions_info_dict = self._initialize_questions_info_dict()

        self.parsing_errors = []  # Errors caused during the information extraction process of the sql queries

        sql_extractor = MoQueryExtractor()

        # Populate dictionary with the datapoints information
        for datapoint_index, datapoint in tqdm(enumerate(dataset),
                                               desc=f"Extracting {self.name} dataset information..."):
            # Get the database information
            datapoint_db_path = datapoint["db_path"] if "db_path" in datapoint else None
            datapoint_db_id = datapoint["db_id"] if "db_id" in datapoint else None
            # If the database path is not None and it does not already exist in the databases dictionary
            if datapoint_db_path is not None and datapoint_db_path not in self._dbs_info_dict["database_path"]:
                if "databases" not in self.ignore_info:
                    try:
                        db_info = DatabaseInfo(db_path_or_name=datapoint_db_path, db_id=datapoint_db_id)
                    except ValueError as e:
                        logger.warning(
                            f"Error in parsing database schema! The database {datapoint_db_path} information will not be"
                            f" considered.")
                        db_info = None

                    self._append_dbs_info_row(datapoint_db_id, datapoint_db_path, db_info)

            # Get the sql query information

            if "sql_queries" not in self.ignore_info:
                try:
                    query_info = sql_extractor.extract(datapoint["sql_query"])
                    # Update dataframe with the information about the query.
                    # (Multiple rows may be created for one sql query - a row can be a subquery)
                    self._append_queries_info_rows(index=datapoint_index, sql_query=datapoint["sql_query"],
                                                   sql_query_info=query_info, db_path=datapoint_db_path,
                                                   store_sql_queries_info=store_sql_queries_info)
                # If there is an error caused by the parser
                except SQLParserException as e:
                    query_info = None
                    self.parsing_errors.append(
                        {"error": f"Error from parser: {e}", "sql_query": datapoint["sql_query"]})
                    # Add a row in the sql queries dict with None in the information from parsing
                    self._append_queries_info_row(index=datapoint_index, sql_query=datapoint["sql_query"],
                                                  sql_query_info=None, db_path=datapoint_db_path,
                                                  store_sql_queries_info=store_sql_queries_info)
                # If there is any other error caused in the query information extraction process
                except Exception as e:
                    self.parsing_errors.append({"error": f"Error at extracting query info: {e}",
                                                "sql_query": datapoint["sql_query"]})
                    query_info = None
                    # Add a row in the sql queries dict with None in the information from parsing
                    self._append_queries_info_row(index=datapoint_index, sql_query=datapoint["sql_query"],
                                                  sql_query_info=None, db_path=datapoint_db_path,
                                                  store_sql_queries_info=store_sql_queries_info)
            else:
                query_info = None

            if "questions" not in self.ignore_info:
                #  Get the natural language question information
                # Add a row in the question dict with no information about the sql query elements
                self._append_question_info_row(index=datapoint_index, question=datapoint["question"],
                                               database_path=datapoint_db_path, sql_query_info=query_info)

        parsing_errors_file_path = f"{PARSING_ERRORS_DIR}/{self.name}_parsing_errors.json"

        if len(self.parsing_errors) > 0:
            # Save to a file the errors during parsing
            with open(parsing_errors_file_path, "w") as error_file:
                error_file.write(json.dumps(self.parsing_errors))

        if len(dataset):
            print(f"Not considered queries: {len(self.parsing_errors)} out of {len(dataset)}")
            if len(self.parsing_errors) > 0:
                print(f"Errors from parsing can be found at {parsing_errors_file_path}")
            print(".......................................................................")

        # Convert the dictionaries with the collected information into a dataframe for quicker extraction of statistics
        self.queries_info_df = pd.DataFrame(self._queries_info_dict)
        self._queries_info_dict.clear()

        self.dbs_info_df = pd.DataFrame(self._dbs_info_dict)
        self._dbs_info_dict.clear()

        self.questions_info_df = pd.DataFrame(self._questions_info_dict)
        self._questions_info_dict.clear()

        if save:
            self.to_files(save_dir=f"{DATASET_INFO_SAVE_DIR}/{self.name}")

    def to_files(self, save_dir: str) -> None:

        Path(f"{save_dir}").mkdir(parents=True, exist_ok=True)

        if "sql_queries" not in self.ignore_info:
            self.queries_info_df.to_pickle(f"{save_dir}/queries_info_df.pkl")
        if "databases" not in self.ignore_info:
            self.dbs_info_df.to_pickle(f"{save_dir}/dbs_info_df.pkl")
        if "questions" not in self.ignore_info:
            self.questions_info_df.to_pickle(f"{save_dir}/questions_info_df.pkl")

        with open(f"{save_dir}/parameters.json", "w") as f:
            f.write(json.dumps({
                "name": self.name,
                "operators": self.operators,
                "variable_types": self.variable_types
            }))

    @classmethod
    def load_from_files(cls, files_dir: str) -> 'DatasetInfo':

        with open(f"{files_dir}/parameters.json", "r") as f:
            parameters = json.load(f)

        dataset_info = cls(dataset_name=parameters["name"], dataset=[])

        dataset_info.operators = parameters["operators"]
        dataset_info.variable_types = parameters["variable_types"]
        dataset_info.queries_info_df = pd.read_pickle(f"{files_dir}/queries_info_df.pkl")
        dataset_info.dbs_info_df = pd.read_pickle(f"{files_dir}/dbs_info_df.pkl")
        dataset_info.questions_info_df = pd.read_pickle(f"{files_dir}/questions_info_df.pkl")

        return dataset_info

    def _initialize_queries_info_dict(self, store_sql_queries_info: bool = False) -> dict:
        """
        Defines the columns of the stored information regarding the sql and the natural language queries.
            store_sql_queries_info (bool): If True the returned dictionary will addionally have the
                'sql_query_info' key.

        Return (dict): Dictionary with keys: names of info to be stored, values: empty lists
        """

        columns = {"index": [], "sql_query": [], "structural_category_name": [],
                   "operatorTypes_category_name": [], "structural_category_counter": [],
                   "operatorTypes_category_counter": [], "select_columns": [], "where_conditions": [],
                   "sql_query_depth": [], "current_depth": [], "db_path": []}

        if store_sql_queries_info:
            columns.update({"sql_query_info": []})

        # Add operators to columns

        for op in self.operators:
            columns[op] = []

        # Add clauses to columns
        for clause in CLAUSES:
            columns[clause.abbr()] = []

        # # Add variable types to columns
        for variable_type in self.variable_types:
            columns[variable_type] = []

        return columns

    @staticmethod
    def _initialize_dbs_info_dict() -> dict:
        """
        Defines the columns of the stored information regarding the databases.
        Returns (dict): Dictionary with keys: names of info to be stored, values: empty lists
        """

        columns = {"database_path": [], "database_id": [], "tables_num": [], "columns_num": [],
                   "foreign_primary_key_relations_num": [], "total_rows": [], "schema_elements": [],
                   "explainable_schema_elements_percentage": []}

        return columns

    @staticmethod
    def _initialize_questions_info_dict() -> dict:
        """
        Defined the columns of the stored information regarding the natural language questions.
        Returns (dict): Dictionary with keys: names of info to be stored, values: empty lists
        """

        columns = {"index": [], "question": [], "length": [], "referenced_schema_elements": [],
                   "sql_query_schema_references": [], "referenced_schema_elements_percentage": [],
                   "readability(flesch_reading_ease)": [], "readability(mcalpine_eflaw)": [],
                   "rarity": [], "lexical_density": [], "avg_dependency_distance": [], "depth": [],
                   "dependencies_num": []}

        return columns

    def _append_queries_info_row(self, index: int, sql_query: Union[str, None], sql_query_info: Union[QueryInfo, None],
                                 db_path: Union[str, None], depth: int = -1,
                                 store_sql_queries_info: bool = False) -> None:
        """
        Adds a new entry in the information dictionary.
        Args:
            index (str): The index of the datapoint
            sql_query (str): The SQL query. If depth != -1 the sql query is None.
            sql_query_info (QueryInfo): The sql query in the form of a QueryInfo class. If the sql query could not be
                parsed the sql_query_info is None.
            db_path (Union[str, None]): The  database upon which the question are made.
                                                 If None, no information about the database are provided.
            depth (int): The depth of the query, if the query is a subquery, else -1.
            store_sql_queries_info (bool): If true the sql_query_info will be saved in the sql queries dictionary
        """

        self._queries_info_dict["index"].append(index)
        self._queries_info_dict["sql_query"].append(sql_query)
        self._queries_info_dict["db_path"].append(db_path)
        self._queries_info_dict["current_depth"].append(depth)

        if store_sql_queries_info:
            self._queries_info_dict["sql_query_info"].append(sql_query_info)

        if sql_query_info is not None:
            self._queries_info_dict["structural_category_name"].append(
                sql_query_info.get_structural_category(return_format="name"))

            operatorTypes_category_name = sql_query_info.get_operatorTypes_category(return_format="name")
            self._queries_info_dict["operatorTypes_category_name"].append(
                operatorTypes_category_name if len(operatorTypes_category_name) > 0 else "None")
            self._queries_info_dict["structural_category_counter"].append(
                sql_query_info.get_structural_category(return_format="counter"))
            self._queries_info_dict["operatorTypes_category_counter"].append(
                sql_query_info.get_operatorTypes_category(return_format="counter"))
            self._queries_info_dict["sql_query_depth"].append(sql_query_info.depth())

            self._queries_info_dict["select_columns"].append(sql_query_info.selectClause.columns_num()
                                                             if sql_query_info.selectClause is not None else None)
            self._queries_info_dict["where_conditions"].append(sql_query_info.whereClause.conditions_num()
                                                               if sql_query_info.whereClause is not None else None)

            query_operators = sql_query_info.operators()
            for operator in self.operators:
                self._queries_info_dict[operator].append(query_operators.count(operator))

            structural_category = sql_query_info.get_structural_category(return_format='counter').split('.')
            for clause_idx, clause in enumerate(CLAUSES):
                self._queries_info_dict[clause.abbr()].append(int(structural_category[clause_idx]))

            for variable_type in self.variable_types:
                try:
                    self._queries_info_dict[variable_type].append(len(getattr(sql_query_info, variable_type)()))
                except Exception as e:
                    # If depth is -1 log the error. Otherwise, the error is already logged in the upper depth of the
                    # sql query
                    if depth == -1:
                        self.parsing_errors.append({"error": f"Error at getting variables of type {variable_type}: e",
                                                    "sql_query": sql_query})
                    self._queries_info_dict[variable_type].append(None)
        else:
            self._queries_info_dict["structural_category_name"].append(None)
            self._queries_info_dict["operatorTypes_category_name"].append(None)
            self._queries_info_dict["structural_category_counter"].append(None)
            self._queries_info_dict["operatorTypes_category_counter"].append(None)
            self._queries_info_dict["sql_query_depth"].append(None)
            self._queries_info_dict["select_columns"].append(None)
            self._queries_info_dict["where_conditions"].append(None)

            for operator in self.operators:
                self._queries_info_dict[operator].append(None)

            for clause_idx, clause in enumerate(CLAUSES):
                self._queries_info_dict[clause.abbr()].append(None)

            for variable_type in self.variable_types:
                self._queries_info_dict[variable_type].append(None)

    def _append_queries_info_rows(self, index: int, sql_query: str, sql_query_info: QueryInfo,
                                  db_path: Union[str, None], store_sql_queries_info: bool = False) -> None:
        """
        Updates the class's information dictionary with the information of the given query.
        Args:
            index (str): The index of the datapoint
            sql_query (str): The SQL query
            sql_query_info (QueryInfo): The sql query in the form of a QueryInfo class
            db_path (Union[str, None]): The information of the database upon which the question are made.
                                                  If None, no information about the database are provided
            store_sql_queries_info (bool): If True the sql_query_info will be stored in the sql queries dictionary
        """

        # Append the query to the information dictionary
        self._append_queries_info_row(index=index, sql_query=sql_query, sql_query_info=sql_query_info, db_path=db_path,
                                      store_sql_queries_info=store_sql_queries_info)

        # Append every subquery to the information dictionary
        queries_per_depth = get_subqueries_per_depth(sql_query_info)

        # For every depth
        for depth, depth_queries in queries_per_depth.items():
            # For every query in the current depth
            for depth_query in depth_queries:
                self._append_queries_info_row(index=index, sql_query=None, sql_query_info=depth_query, db_path=db_path,
                                              depth=depth, store_sql_queries_info=store_sql_queries_info)

    def _append_dbs_info_row(self, database_id: str, database_path: str, db_info: DatabaseInfo = None) -> None:
        """"
        Updates the databases information dictionary with the information of the given database.
        Args:
            database_id (str): The id of the database.
            database_path (str): The path of the database.
            db_info (DatabaseInfo): The class with the information about a database.
        """
        self._dbs_info_dict["database_path"].append(database_path)
        self._dbs_info_dict["database_id"].append(database_id)

        if db_info is not None:
            self._dbs_info_dict["tables_num"].append(db_info.get_tables_num())
            self._dbs_info_dict["columns_num"].append(db_info.get_columns_num())
            self._dbs_info_dict["foreign_primary_key_relations_num"].append(db_info.get_foreign_primary_keys_num())
            self._dbs_info_dict["schema_elements"].append(db_info.get_schema_elements())
            self._dbs_info_dict["explainable_schema_elements_percentage"].append(
                db_info.percentages_of_explainable_schema_elements())
            self._dbs_info_dict["total_rows"].append(db_info.get_total_rows())
        else:
            # Fill the database information with None
            for c in self._dbs_info_dict.keys():
                if c not in ["database_path", "database_id"]:
                    self._dbs_info_dict[c].append(None)

    def _append_question_info_row(self, index: int, question: str, database_path: Union[str, None],
                                  sql_query_info: Union[QueryInfo, None]) -> None:
        """
        Updates the questions information dictionary with the information of the given question.
        Args:
            index (str): The index of the datapoint
            question (str): The natural language question
            database_path (str): The path of the database upon which the question is made
            sql_query_info (QueryInfo): The information of the corresponding sql query
        """

        # Get the schema elements
        schema_elements = self._dbs_info_dict["schema_elements"][
            self._dbs_info_dict["database_path"].index(database_path)] \
            if database_path in self._dbs_info_dict["database_path"] else None

        # Get the schema elements existing in the sql query
        if sql_query_info is not None:
            sql_query_schema_elements = sql_query_info.tables(unique=True)
            sql_query_schema_elements.extend(sql_query_info.columns(return_format="name", unique=True))

        else:
            sql_query_schema_elements = None

        question_info = QuestionInfo(question=question, schema_elements=schema_elements,
                                     sql_query_schema_elements=sql_query_schema_elements)

        self._questions_info_dict["index"].append(index)
        self._questions_info_dict["question"].append(question)
        self._questions_info_dict["length"].append(question_info.question_len())
        self._questions_info_dict["referenced_schema_elements"].append(question_info.referenced_schema_elements())
        self._questions_info_dict["referenced_schema_elements_percentage"].append(
            question_info.referenced_schema_elements_percentage())
        self._questions_info_dict["sql_query_schema_references"].append(question_info.sql_query_schema_elements)
        self._questions_info_dict["readability(flesch_reading_ease)"].append(
            question_info.readability(metric="flesch_reading_ease"))
        self._questions_info_dict["readability(mcalpine_eflaw)"].append(
            question_info.readability(metric="mcalpine_eflaw"))
        self._questions_info_dict["rarity"].append(question_info.rarity())
        self._questions_info_dict["lexical_density"].append(question_info.lexical_density())
        self._questions_info_dict["avg_dependency_distance"].append(question_info.avg_dependency_distance())
        self._questions_info_dict["depth"].append(question_info.depth())
        self._questions_info_dict["dependencies_num"].append(question_info.dependencies_num())

    def _most_common_categories(self, categories_percentages: pd.DataFrame, dataset_max_depth: int, categ_num: int,
                                keep_indexes: bool = False) -> pd.DataFrame:
        """
        Returns the most common categories of the given data and adds an extra category named "Other" with the
        percentage of the rest.

        *The most common categories are produced without considering the per depth percentages.

        Args:
            categories_percentages: DataFrame with the columns {<category_name>, "Depth", "% Queries", Optional("Indexes")}
            dataset_max_depth (int): The maximum depth in the dataset
            categ_num (int): The number of the common categories to keep.
            keep_indexes (bool): If True, queries indexes of the 'Other' category are kept in an extra column named
                                'indexes'.

            !!! Currently the most common categories are available only for a unique depth.
        """
        category_name = categories_percentages.columns[0]

        # Get the most common categories names (no depth considered)
        most_common_categories = list(
            categories_percentages[categories_percentages["Depth"] == -1].sort_values(by=["% Queries"], ascending=False)
            [category_name].head(categ_num).values)

        # Remove the categories not included in the most common
        categories_percentages = categories_percentages[categories_percentages[category_name]. \
            isin(most_common_categories)]

        # Add 'other' row with the percentages of dismissed categories for each depth
        other_row = {category_name: "Other",
                     "Depth": -1,
                     "% Queries": 100 - categories_percentages[categories_percentages["Depth"] == -1][
                         "% Queries"].sum()
                     }

        if keep_indexes:
            other_row["Indexes"] = self.queries_info_df[(self.queries_info_df["current_depth"] == -1) &
                                                        (~self.queries_info_df.index.isin(
                                                            [index for index_set in
                                                             categories_percentages["Indexes"].tolist() for index in
                                                             index_set]))].index.values
        categories_percentages = categories_percentages.append(other_row, ignore_index=True)

        return categories_percentages

    def _categoriesDf(self, categorization_name: str, categ_num: int = None, max_depth: int = None,
                      keep_indexes: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the percentages of the categories per depth.
        Args:
            categorization_name (str): The name of the categorization in the self.queries_info_df
            categ_num (int): The number of the categories returned. If None all categories are
                             returned, else the <categ_num> most common categories are returned, with a extra category
                             'Other' that contains the rest of the categories.
            max_depth (int): The max depth that we want to consider. By default, depth is None which means that we want
                             the categories of the sql queries for all depths.
            keep_indexes (bool): If True the sql queries indexes for each category are kept.

        Returns (DataFrame): A dataframe containing the columns = [<categorization_name>(str), 'Depth'(int),
                                                                   '% Queries' (int), Optional('Indexes'(list))].
        """

        # Initialize the categories dataframe
        categories = {categorization_name: [], "Depth": [], "% Queries": []}

        if keep_indexes:
            categories["Indexes"] = []

        # If max depth has not defined, set max_depth to the maximum depth of the dataset
        if max_depth is None:
            max_depth = self.queries_info_df["sql_query_depth"].max()

        # Remove all rows in the dataframe without information about the sql_query
        queries = self.queries_info_df[self.queries_info_df[categorization_name].notnull()]

        # Group the dataframe with the dataset's information by depth
        grouped_depths = queries[queries["current_depth"] <= max_depth].groupby("current_depth")

        # Create a list with the names of the unique categories
        all_categories = queries[categorization_name].unique()

        # For every depth
        for depth, depth_queries in grouped_depths:

            # Get the structural categories and their count for the current depth
            depth_categories = depth_queries[categorization_name].value_counts()

            # Get the number of queries in the current depth
            sql_queries_num_in_current_depth = depth_queries.shape[0]

            # For every category
            for category in all_categories:

                # Get the number of queries in the structural category
                category_sql_queries_num = depth_categories[category] \
                    if category in depth_categories else 0

                # Update the categories dataframe
                categories[categorization_name].append(category)
                categories["Depth"].append(depth)
                categories["% Queries"].append(100.0 * category_sql_queries_num /
                                               sql_queries_num_in_current_depth)
                if keep_indexes:
                    categories["Indexes"].append(depth_queries[depth_queries[categorization_name] ==
                                                               category]["index"].values)

        categories_df = pd.DataFrame(categories)

        # If we want only the most common categories
        if categ_num is not None:
            max_depth = max(list(grouped_depths.groups.keys()))
            categories_df = self._most_common_categories(categories_df, max_depth, categ_num, keep_indexes)

        # Add the name of the dataset in the dataframe with the categories
        categories_df["Dataset"] = self.name
        return categories_df

    def _get_statistics(self, values: pd.Series, extended: bool = False) -> \
            Union[tuple[float, int, int], dict]:
        """
        Calculates the summary statistics of the given pd.Series.
        Args:
            values: The given values.
            extended: If False the (mean, min, max) will be returned. Otherwise, the matplotlib.cbook.boxplot_stats will
                be returned.
        """
        if extended:
            return cbook.boxplot_stats(values.to_numpy(), labels=[self.name])[0]
        else:
            return round(values.mean(), 2), values.min(), values.max()

    def structural_categories_df(self, categ_num: int = None, max_depth: int = None, keep_indexes: bool = False,
                                 grouped: bool = True) -> pd.DataFrame:
        """
        Creates a DataFrame with the percentages of the structural categories per depth.
        Args:
            categ_num (int): The number of the structural categories returned. If None all structural types are
                             returned, else the <categ_num> most common categories are returned.
            max_depth (int): The max depth that we want to consider. By default, depth is None which means that we want
                             the structural categories of the queries for all depths.
            keep_indexes (bool): If True the queries indexes for each category are kept.
            grouped (bool): If True the same number of structural categories are grouped and a column named '% Queries',
                with the percentage of queries in the dataset of each unique structural category is added in the
                dataframe.

        Returns (DataFrame): A dataframe containing the columns = ['Structural Category'(str), 'Depth'(int),
        '% Queries' (int), 'Dataset' (str), Optional('Indexes'(list))].
        """

        if grouped:
            return self._categoriesDf(categorization_name="structural_category_name", categ_num=categ_num,
                                      max_depth=max_depth,
                                      keep_indexes=keep_indexes)
        else:
            df = self.queries_info_df[self.queries_info_df["current_depth"] <= max_depth][["structural_category_name",
                                                                                           "current_depth"]]
            df.rename(columns={"structural_category_name": "Structural category", "current_depth": "Depth"},
                      inplace=True)
            df["Dataset"] = self.name

            return df

    def operatorTypes_categories_df(self, categ_num: int = None, max_depth: int = None, keep_indexes: bool = False,
                                    grouped: bool = True) -> pd.DataFrame:
        """
        Returns a Dataframe with the percentages of the operator types categories per depth.
        Args:
            categ_num (int): The number of the structural categories returned. If None all structural types are
                 returned, else the <categ_num> most common categories are returned.
            max_depth (int): The max depth that we want to consider. By default, depth is None which means that we want
                             the structural categories of the queries for all depths.
            keep_indexes (bool): If True the queries indexes for each category are kept.
            grouped (bool): If True the same number of operator type categories are grouped and a column named
                '% Queries', with the percentage of queries in the dataset of each unique operator type category is
                added in the dataframe.

        Returns (DataFrame): A dataframe containing the columns = ['Operator Types Category'(str), 'Depth'(int),
        '% Queries' (int), 'Dataset' (str), Optional('Indexes'(list))].
        """
        if grouped:
            return self._categoriesDf(categorization_name="operatorTypes_category_name", categ_num=categ_num,
                                      max_depth=max_depth, keep_indexes=keep_indexes)
        else:
            df = self.queries_info_df[self.queries_info_df["current_depth"] <= max_depth][
                ["operatorTypes_category_name", "current_depth"]]
            df.rename(columns={"operatorTypes_category_name": "Operator category", "current_depth": "Depth"},
                      inplace=True)
            df["Dataset"] = self.name

            return df

    def operators_num(self, unique: bool = False, grouped: bool = True, add_dataset_name: bool = True) -> pd.DataFrame:
        """
        Returns a dataframe with the percentages of the operator number in the dataset's queries.
        Args:
            unique (bool): If True the number of unique operators in a query will be considered, otherwise the number of
                all operators in the query is considered.
            grouped (bool): If True the same number of operators are grouped and a column named '% Queries', with the
                percentage of queries in the dataset of each unique operators number is added in the dataframe.
            add_dataset_name (bool): If True a column named 'Dataset' with the name of the dataset will be added in the
                dataframe.

        Returns: A dataframe containing the columns = ['# Operators' | '# Unique Operators' (int), 'Depth'(int),
        '% Queries'(int), 'Dataset' (str)]


        !!! Currently, only depth = -1 is supported.
        """

        # Get the queries of the dataset (no depth considered) for which there are information about the operators
        queries = self.queries_info_df[self.queries_info_df[self.operators[0]].notnull()]
        queries = queries.groupby("current_depth").get_group(-1)

        # Get the number of operators in the queries
        if unique:
            operators_num_in_queries = queries[self.operators].gt(0).sum(axis=1)
        else:
            operators_num_in_queries = queries[self.operators].sum(axis=1)

        if grouped:
            # Create a dataframe with the percentages of the operators numbers in the queries
            ops_num = {f"# {'Unique ' if unique else ''}Operators": [], "Depth": [], "% Queries": []}
            for operators_num, queries_num in operators_num_in_queries.value_counts().items():
                ops_num[f"# {'Unique ' if unique else ''}Operators"].append(int(operators_num))
                ops_num["Depth"].append(-1)
                ops_num["% Queries"].append(queries_num / float(queries.shape[0]) * 100)

            ops_num_df = pd.DataFrame(ops_num)
        else:
            ops_num_df = pd.DataFrame({"# Operators": operators_num_in_queries,
                                       "Depth": [-1 for _ in range(operators_num_in_queries.shape[0])]})

        if add_dataset_name:
            ops_num_df["Dataset"] = self.name

        return ops_num_df

    def operator_types_num(self) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages of the operator types number in the dataset's queries.

        Returns: A dataframe containing the columns = ['# Operator Types'(str), 'Depth'(int), '% Queries'(int)]
        """

        # Get the queries of the dataset (no depth considered) for which there are information about the operators
        queries = self.queries_info_df[self.queries_info_df[self.operators[0]].notnull()]
        queries = queries.groupby("current_depth").get_group(-1)

        # Get the number of each operator type in all the queries
        queries_op_types = pd.DataFrame({})
        for operator_type in OPERATOR_TYPES:
            queries_op_types[operator_type.abbr()] = queries[operator_type.abbr()].sum(axis=1).gt(0)

        # Get the number of operator types in the queries
        op_types_num_in_queries = queries_op_types.sum(axis=1)

        optypes_num = {"# Operator Types": [], "Depth": [], "% Queries": []}
        for operatorTypes_num, queries_num in op_types_num_in_queries.value_counts.items():
            optypes_num["# Operator Types"].append(int(operatorTypes_num))
            optypes_num["Depth"].append(-1)
            optypes_num["% Queries"].append(queries_num / queries.shape[0] * 100)

        optypes_num = pd.DataFrame(optypes_num)
        optypes_num["Dataset"] = self.name

        return optypes_num

    def _variable_type_df(self, variable_type: str, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages of the variable type number in the dataset's sql queries.

        Args:
            variable_type (str): The name of the variable type for which the dataframe will be created.
            grouped (bool): If True the same number of <variable_type> are grouped and a column named '% Queries', with
                the percentage of queries in the dataset of each unique <variable_type> number is added in the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['# <variable_type>', 'Depth'(int), '% Queries'(int), 'Dataset'(str)]

        !!! Currently, only depth = -1 is supported.
        """

        if variable_type not in self.variable_types:
            raise ValueError(f"{variable_type} is not a valid variable type!")

        # Get the queries of the dataset (no depth considered) for which there are information about the operators
        queries = self.queries_info_df[self.queries_info_df[variable_type].notnull()]

        if grouped:
            queries = queries.groupby("current_depth").get_group(-1)

            variable_type_df = {f"# {variable_type}": [], "Depth": [], "% Queries": []}
            for variable_type_num, queries_num in queries[variable_type].value_counts().items():
                variable_type_df[f"# {variable_type}"].append(int(variable_type_num))
                variable_type_df["Depth"].append(-1)
                variable_type_df["% Queries"].append(queries_num / queries.shape[0] * 100)

            variable_type_df = pd.DataFrame(variable_type_df)
        else:
            variable_type_df = queries[queries["current_depth"] == -1][[variable_type, "current_depth"]]
            variable_type_df = variable_type_df.rename(columns={variable_type: f"# {variable_type}",
                                                                'current_depth': "Depth"})

        variable_type_df["Dataset"] = self.name

        return variable_type_df

    def sql_queries_tables_num(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages of the tables number in the dataset's sql queries.

        Args:
            grouped (bool): If True the same number of tables are grouped and a column named '% Queries', with
                the percentage of queries in the dataset of each unique table number is added in the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['# tables', 'Depth'(int), '% Queries'(int), 'Dataset']
        """
        return self._variable_type_df(variable_type="tables", grouped=grouped)

    def sql_queries_tables_num_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """Returns the summary statistics of number of tables used in the sql queries."""

        tables_num = self._variable_type_df(variable_type="tables")["# tables"].dropna()
        return self._get_statistics(tables_num, extended=extended)

    def sql_queries_columns_num(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages of the columns number in the dataset's sql queries.

        Args:
            grouped (bool): If True the same number of columns are grouped and a column named '% Queries', with
                the percentage of queries in the dataset of each unique column number is added in the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['# columns', 'Depth'(int), '% Queries'(int), 'Dataset']
        """
        return self._variable_type_df(variable_type="columns", grouped=grouped)

    def sql_queries_columns_num_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """Returns the summary statistics of number of columns used in the sql queries."""
        columns_num = self._variable_type_df(variable_type="columns")["# columns"].dropna()
        return self._get_statistics(columns_num, extended=extended)

    def sql_queries_values_num(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages of the values number in the dataset's sql queries.

        Args:
            grouped (bool): If True the same number of columns are grouped and a column named '% Queries', with
                the percentage of queries in the dataset of each unique column number is added in the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['# values', 'Depth'(int), '% Queries'(int), 'Dataset']
        """
        return self._variable_type_df(variable_type="values", grouped=grouped)

    def unique_sql_queries_num(self) -> int:
        """Returns the number of the unique sql queries in the dataset"""
        return self.queries_info_df["sql_query"].nunique()

    def unique_questions_num(self) -> int:
        """Returns the number of unique natural language questions in the dataset"""
        return self.questions_info_df["question"].nunique()

    def templates_num(self, template_type: Literal["counter", "name"]) -> int:
        """Returns the number of templates (structural category + operatorTypes category) in the dataset"""
        template = (self.queries_info_df[f"structural_category_{template_type}"] +
                    self.queries_info_df[f"operatorTypes_category_{template_type}"])
        return template.nunique()

    def unique_question_sql_queries_num(self) -> int:
        """Returns the number of unique sql, nlq pairs in the dataset"""
        questions_and_sql_queries = pd.merge(self.questions_info_df[["question", "index"]],
                                             self.queries_info_df[self.queries_info_df["current_depth"] == -1][
                                                 ["sql_query", "index"]],
                                             on="index")
        pairs = questions_and_sql_queries["sql_query"] + " " + questions_and_sql_queries["question"]
        return pairs.nunique()

    def sql_queries_depth_statistics(self) -> (float, int, int):
        """Returns the average, the minimum and the maximum depth of the sql queries in the dataset"""
        # Get all the depths of all the queries of the dataset
        sql_queries_depths = self.queries_info_df[self.queries_info_df["current_depth"] == -1][
            "sql_query_depth"].dropna()

        return round(sql_queries_depths.mean(), 2), sql_queries_depths.min(), sql_queries_depths.max()

    def sql_queries_operators_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """Returns the summary statistics of operators in the sql queries in the dataset"""
        operators_num = self.operators_num(grouped=False, add_dataset_name=False)["# Operators"]
        return self._get_statistics(operators_num, extended=extended)

    def sql_queries_values_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """Returns the summary statistics of operators in the sql queries in the dataset"""

        values_counter = self.queries_info_df[self.queries_info_df["current_depth"] == -1][
            "values"].dropna()
        return self._get_statistics(values_counter, extended=extended)

    def sql_queries_joins_statistics(self) -> (float, int, int):
        """Returns the average, the minimum and the maximum number of joins in the sql queries in the dataset"""
        # Get the operator categories (in 'counter' format) of all the queries of the dataset
        operator_types_counter = self.queries_info_df[self.queries_info_df["current_depth"] == -1][
            "operatorTypes_category_counter"].dropna()

        # Keep only the information about the join operator type
        joins_counter = operator_types_counter.map(lambda op_type: int(op_type.split(".")[0]))

        return round(joins_counter.mean(), 2), joins_counter.min(), joins_counter.max()

    def sql_queries_select_columns_statistics(self) -> (float, int, int):
        """
        Returns the average, the minimum and the maximum number of columns in the base select clause in the queries
        of the dataset.
        """
        # Get all the queries of the dataset with information about the select clause
        select_columns = self.queries_info_df[self.queries_info_df["current_depth"] == -1][
            "select_columns"].dropna()

        return round(select_columns.mean(), 2), select_columns.min(), select_columns.max()

    def sql_queries_where_conditions_statistics(self) -> (float, int, int):
        """
        Returns the average, the minimum and the maximum number of conditions in the base where clause in the
        queries of the dataset.
        """
        # Get all the queries of the dataset with information about the select clause
        where_conditions = self.queries_info_df[self.queries_info_df["current_depth"] == -1][
            "where_conditions"].dropna()

        return round(where_conditions.mean(), 2), where_conditions.min(), where_conditions.max()

    def databases_num(self) -> int:
        """Returns the number of databases in the dataset"""
        return self.dbs_info_df.shape[0]

    def _get_attr_percentages(self, df: pd.DataFrame, attr_name: str, count_name: str) -> pd.DataFrame:
        """
        Returns a dataframe with the percentage of each attribute value in the given dataframe.

        Args:
            df (pd.DataFrame): Thr dataframe that contains the attribute
            attr_name (str): The name of attribute for which we want to calculate the percentage of each value
            count_name (str): The name of the value counted
        """
        # Get the rows of the attribute for which there is information
        attr_values = df[df[attr_name].notnull()][attr_name]

        rows_num = attr_values.shape[0]
        if rows_num != 0:
            attr_num_percentages = dict(map(lambda item: (item[0], item[1] / rows_num * 100.0),
                                            attr_values.value_counts().items()))
            df = pd.DataFrame({f"% {count_name}": attr_num_percentages.values(),
                               attr_name: attr_num_percentages.keys()})
        else:
            df = pd.DataFrame({f"% {count_name}": [], attr_name: []})

        df["Dataset"] = self.name
        return df

    def dbs_tables_num(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages per tables number existing in the dataset's databases.
        Args:
            grouped (bool): If True the same number of tables are grouped and a column named
                '% Databases', with the percentage of databases in the dataset of each unique number of tables
                is added in the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['# Tables', '% Databases', 'Dataset']
        """

        if grouped:
            df = self._get_attr_percentages(self.dbs_info_df, "tables_num", "Databases")
        else:
            df = self.dbs_info_df[self.dbs_info_df["tables_num"].notnull()][["tables_num"]]
            df["Dataset"] = self.name

        df.rename(columns={"tables_num": "# Tables"}, inplace=True)
        return df

    def dbs_columns_num(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages per columns number existing in the dataset's databases.
        Args:
            grouped (bool): If True the same number of columns are grouped and a column named
                '% Databases', with the percentage of databases in the dataset of each unique number of columns
                is added in the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['# Columns', '% Databases', 'Dataset']
        """
        if grouped:
            df = self._get_attr_percentages(self.dbs_info_df, "columns_num", "Databases")
        else:
            df = self.dbs_info_df[self.dbs_info_df["columns_num"].notnull()][["columns_num"]]
            df["Dataset"] = self.name

        df.rename(columns={"columns_num": "# Columns"}, inplace=True)
        return df

    def dbs_foreign_primary_key_relations_num(self) -> pd.DataFrame:
        """Returns a dataframe with the percentages per fp-key relations number existing in the dataset's databases."""
        df = self._get_attr_percentages(self.dbs_info_df, "foreign_primary_key_relations_num", "Databases")
        df.rename(columns={"foreign_primary_key_relations_num": "# Fp relations"}, inplace=True)
        return df

    def dbs_explainable_schema_elements_percentage(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages per percentage of explainable schema elements in the dataset's
        databases.

        Args:
            grouped (bool): If True the same percentages of explainable schema elements are grouped and a column named
                '% Databases', with the percentage of databases per explainable schema elements percentage is added in
                the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['% Explainable schema elements', '% Databases', 'Dataset']
        """
        if grouped:
            df = self._get_attr_percentages(self.dbs_info_df, "explainable_schema_elements_percentage",
                                            "Databases")
        else:
            df = self.dbs_info_df[self.dbs_info_df["explainable_schema_elements_percentage"].notnull()][
                ["explainable_schema_elements_percentage"]]
            df["Dataset"] = self.name

        df.rename(columns={"explainable_schema_elements_percentage": "% Explainable schema elements"}, inplace=True)
        return df

    def dbs_explainable_schema_elements_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """
        Returns the summary statistics of the percentages of explainable schema elements in the databases.
        """
        expl_schema_elements_perc = self.dbs_info_df["explainable_schema_elements_percentage"].dropna()
        return self._get_statistics(expl_schema_elements_perc, extended=extended)

    def dbs_total_rows_df(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages per total rows number existing in the dataset's
        databases.
        Args:
            grouped (bool): If True the databases with the same total number of rows are grouped and a column named
                '% Databases', with the databases' rows, is added in the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['# Rows', '% Databases', 'Dataset']

        """
        if grouped:
            df = self._get_attr_percentages(self.dbs_info_df, "total_rows", "Databases")
        else:
            df = self.dbs_info_df[self.dbs_info_df["total_rows"].notnull()][["total_rows"]]
            df["Dataset"] = self.name

        df.rename(columns={"total_rows": "# Rows"}, inplace=True)
        return df

    def dbs_tables_rows_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """
        Returns the summary statistics of the table rows in each database.
        """

        table_rows = self.dbs_info_df["average_tables_rows"].dropna()
        return self._get_statistics(table_rows, extended=extended)

    def questions_length_df(self, grouped: bool = True) -> pd.DataFrame:
        """Returns a dataframe with the percentages per question length existing in the dataset."""
        question_lens = self.questions_info_df[self.questions_info_df["length"].notnull()]["length"]

        if grouped:
            df = self._get_attr_percentages(self.questions_info_df, "length", "Questions")
        else:
            df = self.questions_info_df[self.questions_info_df["length"].notnull()][["length"]]
            df["Dataset"] = self.name

        df.rename(columns={"length": "Question Length"}, inplace=True)
        return df

    def questions_length_statistics(self) -> (float, int, int):
        """Returns the average, the minimum and the maximum length of the natural language questions in the dataset."""
        question_lengths = self.questions_info_df["length"].dropna()

        return round(question_lengths.mean(), 2), question_lengths.min(), question_lengths.max()

    def questions_exact_schema_references_df(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the percentages values of the schema elements used in the query and referenced by
        their exact name in the natural language question.

        Args:
            grouped (bool): If True the same percentages of exact schema references are grouped and a column named
                '% Questions', with the percentage of queries in the dataset of each unique exact schema reference
                percentage number is added in the dataframe.

        Returns (pd.DataFrame): A dataframe containing the
            columns = ['% Exact references', '% Queries'(int), 'Dataset']

        """
        if grouped:
            df = self._get_attr_percentages(self.questions_info_df, "referenced_schema_elements_percentage",
                                            "Questions")
        else:
            df = self.questions_info_df[self.questions_info_df["referenced_schema_elements_percentage"].notnull()][[
                "referenced_schema_elements_percentage"]]
            df["Dataset"] = self.name

        df.rename(columns={"referenced_schema_elements_percentage": "% Exact references"}, inplace=True)
        return df

    def questions_exact_schema_references_statistics(self, extended: bool = False) -> Union[
        tuple[float, int, int], dict]:
        """
        Returns the summary statistics of schema elements existing in the sql query and referenced by their exact name
        in the natural language question.
        """
        exact_refs_per = self.questions_info_df["referenced_schema_elements_percentage"].dropna()
        return self._get_statistics(exact_refs_per, extended=extended)

    def questions_readability_df(self,
                                 metric: Literal["flesch_reading_ease", "mcalpine_eflaw"] = "flesch_reading_ease",
                                 grouped: bool = True) \
            -> pd.DataFrame:
        """
        Creates a dataframe with the percentages per readability existing in the dataset.

        Args:
            metric: The metric used for the calculation of the readability score
            grouped (bool): If True the same readability scores are grouped and a column named '% Questions', with the
                percentage of queries in the dataset for each unique readability score is added in the dataframe.

        Returns (pd.DataFrame): columns=["Dataset", "% Questions", "Question Readability(<metric>)"]
        """

        if grouped:
            df = self._get_attr_percentages(self.questions_info_df, f"readability({metric})", "Questions")
        else:
            df = self.questions_info_df[self.questions_info_df[f"readability({metric})"].notnull()][[
                f"readability({metric})"]]
            df["Dataset"] = self.name

        df.rename(columns={f'readability({metric})': f"Question Readability"}, inplace=True)

        return df

    def questions_readability_statistics(self, metric: Literal[
        "flesch_reading_ease", "mcalpine_eflaw"] = "flesch_reading_ease",
                                         extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """
        Returns the summary statistics of readability scores in the dataset's questions.
        """
        readability = self.questions_info_df[f"readability({metric})"].dropna()
        return self._get_statistics(readability, extended=extended)

    def questions_variability(self) -> float:
        """Returns the variability score of the dataset's questions."""
        questions = self.questions_info_df["question"].to_list()
        questions = " ".join(questions)

        return text_complexity_measures.surfaced_based_metrics(questions, metrics=['type-token ratio'])[
            "type-token ratio"]

    def questions_rarity_df(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the rarity scores in the dataset's questions.

        Args:
            grouped (bool): If True the same rarity scores are grouped and a column named '% Questions', with the
                percentage of queries in the dataset for each unique rarity score is added in the dataframe.

        Returns(pd.DataFrame): columns=["Dataset", "% Questions", "Question Rarity"]
        """
        if grouped:
            df = self._get_attr_percentages(self.questions_info_df, "rarity", "Questions")
        else:
            df = self.questions_info_df[self.questions_info_df["rarity"].notnull()][["rarity"]]
            df["Dataset"] = self.name

        df.rename(columns={"rarity": "Question Rarity"}, inplace=True)
        return df

    def questions_rarity_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """
        Returns the summary statistics of the rarity scores in the dataset's questions.
        """

        rarity_scores = self.questions_info_df["rarity"].dropna()
        return self._get_statistics(rarity_scores, extended=extended)

    def questions_lexical_density_df(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the lexical density scores in the dataset's questions.

        Args:
            grouped (bool): If True the same lexical density scores are grouped and a column named '% Questions', with
                the percentage of queries in the dataset for each unique lexical density score is added in the dataframe.

        Returns(pd.DataFrame): columns=["Dataset", "% Questions", "Question lexical density"]
        """

        if grouped:
            df = self._get_attr_percentages(self.questions_info_df, "lexical_density", "Questions")
        else:
            df = self.questions_info_df[self.questions_info_df["lexical_density"].notnull()][["lexical_density"]]
            df["Dataset"] = self.name

        df.rename(columns={"lexical_density": "Question lexical density"}, inplace=True)
        return df

    def questions_lexical_density_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """
        Returns the summary statistics of the lexical density scores in the dataset's questions.
        """
        lexical_density_scores = self.questions_info_df["lexical_density"].dropna()
        return self._get_statistics(lexical_density_scores, extended=extended)

    def questions_avg_dependency_distance_df(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the average dependency distance in the dependency trees of the dataset's questions.

        Args:
            grouped (bool): If True the same avg dependency distances are grouped and a column named '% Questions', with
                the percentage of queries in the dataset for each unique avg dependency distance is added in the
                dataframe.

        Returns(pd.DataFrame): columns=["Dataset", "% Questions", "Question avg dependency distance"]
        """
        if grouped:
            df = self._get_attr_percentages(self.questions_info_df, "avg_dependency_distance", "Questions")
        else:
            df = self.questions_info_df[self.questions_info_df["avg_dependency_distance"].notnull()][[
                "avg_dependency_distance"]]
            df["Dataset"] = self.name

        df.rename(columns={"avg_dependency_distance": "Question avg dependency distance"}, inplace=True)
        return df

    def questions_avg_dependency_distance_statistics(self, extended: bool = False) -> \
            Union[tuple[float, int, int], dict]:
        """
        Returns the summary statistics of the avg dependency distance in the dataset's questions.
        """
        avg_dependency_distance = self.questions_info_df["avg_dependency_distance"].dropna()
        return self._get_statistics(avg_dependency_distance, extended=extended)

    def questions_dependencies_depth_df(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the depth of the dependency parse trees in the dependency tree of the dataset's
        questions.

        Args:
            grouped (bool): If True the same dependencies depth are grouped and a column named '% Questions', with
                the percentage of queries in the dataset for each unique dependencies depth is added in the dataframe.

        Returns(pd.DataFrame): columns=["Dataset", "% Questions", "Questions dependencies depth"]
        """
        if grouped:
            df = self._get_attr_percentages(self.questions_info_df, "depth", "Questions")
        else:
            df = self.questions_info_df[self.questions_info_df["depth"].notnull()][["depth"]]
            df["Dataset"] = self.name

        df.rename(columns={"depth": "Question dependencies depth"}, inplace=True)
        return df

    def questions_dependencies_depth_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """
        Returns the summary statistics of the dependency depth in the dataset's questions.
        """
        dependencies_depth = self.questions_info_df["depth"].dropna()
        return self._get_statistics(dependencies_depth, extended=extended)

    def questions_dependencies_num_df(self, grouped: bool = True) -> pd.DataFrame:
        """
        Creates a dataframe with the number of the dependencies in the dependency trees of the dataset's
        questions.

        Args:
            grouped (bool): If True the same dependencies number are grouped and a column named '% Questions', with
                the percentage of queries in the dataset for each unique dependencies number is added in the dataframe.

        Returns(pd.DataFrame): columns=["Dataset", "% Questions", "Questions dependency num"]
        """
        if grouped:
            df = self._get_attr_percentages(self.questions_info_df, "dependencies_num", "Questions")
        else:
            df = self.questions_info_df[self.questions_info_df["dependencies_num"].notnull()][["dependencies_num"]]
            df["Dataset"] = self.name

        df.rename(columns={"dependencies_num": "Question dependencies num"}, inplace=True)
        return df

    def questions_dependencies_num_statistics(self, extended: bool = False) -> Union[tuple[float, int, int], dict]:
        """
        Returns the summary statistics of the number of the dependencies in the dependency trees in the dataset's
        questions.
        """
        dependencies_num = self.questions_info_df["dependencies_num"].dropna()
        return self._get_statistics(dependencies_num, extended=extended)

    def structural_variety(self) -> float:

        structural_categories = self.structural_categories_df(max_depth=-1)
        return sum([(p / 100) * (1 - (p / 100)) for p in structural_categories["% Queries"]])

    def operator_variety(self) -> float:

        operator_categories = self.operatorTypes_categories_df(max_depth=-1)
        return sum([(p / 100) * (1 - (p / 100)) for p in operator_categories["% Queries"]])

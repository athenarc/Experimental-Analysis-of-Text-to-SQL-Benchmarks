from typing import Union
import enchant

from DatasetAnalysisTools.DatabaseInfo.databases.json_database import JsonDatabase
from DatasetAnalysisTools.DatabaseInfo.databases.mysql_database import MySQLDatabase
from DatasetAnalysisTools.DatabaseInfo.databases.sqlite_database import SqliteDatabase

d = enchant.Dict("en")


class DatabaseInfo:

    table_names: list[str]
    column_names: list[tuple[str, str]]  # tuple(table_index, column_name)
    column_types: list[str]  # list with the type of each column
    foreign_primary_keys: list[tuple[str, str]]  # tuple(<foreign_key_column_index>, <primary_key_column_index>)
    table_rows: Union[list[int], None]

    def __init__(self, db_path_or_name: str, **kwargs):
        """
        Initializes the information of a database.
        Args:
            db_path_or_name (str): The path to an .sqlite file, to a .json file, or to a MySQL database containing the
                                   tables names, column names, column types and the foreign primary key relations of the database.
        """
        try:
            # If a .sqlite file has been given to get the database schema
            if db_path_or_name.endswith(".sqlite"):
                db = SqliteDatabase(db_path_or_name)
            elif db_path_or_name.endswith(".json"):
                db = JsonDatabase(db_path_or_name, **kwargs)
            else:
                db = MySQLDatabase(db_path_or_name)
        except Exception as e:
            raise ValueError(f"Database {db_path_or_name} is not valid!")

        (self.table_names, self.column_names,
         self.column_types, self.foreign_primary_keys, self.table_rows) = db.get_schema()

    def schema_as_dict(self) -> dict:
        return {
            "table_names": self.table_names,
            "column_names": self.column_names,
            "column_types": self.column_types,
            "foreign_primary_keys": self.foreign_primary_keys
        }

    def get_columns_per_table(self, lowercase: bool = False) -> dict:
        """
        Returns a dictionary with keyw the table names and values the columns of each table.

        Args:
            lowercase (bool): If True the names of the columns and tables are lowercased.

        """
        columns_per_table = {}
        for t_i, table in enumerate(self.table_names):
            table_columns = list(filter(lambda column: column[0] == t_i, self.column_names))

            columns_per_table[table.lower() if lowercase else table] = list(map(
                lambda column: column[1].lower() if lowercase else column[1], table_columns))

        return columns_per_table

    def get_schema_elements(self) -> list[str]:
        """ Returns the database schema as a list with the names of the columns and the tables in the database. """
        schema_elements = []
        for t_i, table in enumerate(self.table_names):
            # Get the columns of the current table
            table_columns = list(filter(lambda column: column[0] == t_i, self.column_names))
            # Get the columns names and append them to the schema elements
            schema_elements.extend(list(map(lambda column: column[1], table_columns)))

        # Add the tables in the schema elements
        schema_elements.extend(self.table_names)

        return schema_elements

    def get_columns_num(self) -> int:
        """Returns the number of columns existing in the database"""
        # -1 for '*'
        return len(self.column_names) - 1

    def get_tables_num(self) -> int:
        """Returns the number of tables existing in the database"""
        return len(self.table_names)

    def get_foreign_primary_keys_num(self) -> int:
        """Returns the number of foreign primary key relations existing in the database"""
        return len(self.foreign_primary_keys)

    def get_tables_rows_stats(self) -> Union[tuple[float, int, int], None]:
        """Returns the average, min and max number of rows for the tables of the database"""
        if self.table_rows is None:
            return None
        else:
            return sum(self.table_rows) / len(self.table_rows), min(self.table_rows), max(self.table_rows)

    def get_total_rows(self) -> Union[int, None]:
        return sum(self.table_rows) if self.table_rows is not None else None

    def percentages_of_explainable_schema_elements(self) -> float:
        """Returns the percentage of the schema elements that are valid English words."""

        schema_elements = self.get_schema_elements()

        explainable_schema_elements_count = 0
        for schema_element in schema_elements:
            valid_schema_element = True
            for schema_element_word in schema_element.split("_"):
                valid_schema_element = d.check(schema_element_word)
                if not valid_schema_element:
                    break

            if valid_schema_element:
                explainable_schema_elements_count += 1

        return explainable_schema_elements_count / len(schema_elements) * 100

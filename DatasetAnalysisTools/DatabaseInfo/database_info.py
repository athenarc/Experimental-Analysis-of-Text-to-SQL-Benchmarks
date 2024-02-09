import json
import sqlite3
from typing import Union
import enchant

d = enchant.Dict("en")


class DatabaseInfo:

    table_names: list[str]
    column_names: list[tuple[str, str]]  # tuple(table_index, column_name)
    column_types: list[str]  # list with the type of each column
    foreign_primary_keys: list[tuple[str, str]]  # tuple(<foreign_key_column_index>, <primary_key_column_index>)
    table_rows: Union[list[int], None]

    def __init__(self, db_path_or_json: str):
        """
        Initializes the information of a database.
        Args:
            db_path_or_json (str): The path to an .sqlite file or to a .json file containing the tables names, column
                                   names, column types and the foreign primary key relations of the database.
        """
        # If a .sqlite file has been given to get the database schema
        if db_path_or_json.endswith(".sqlite"):
            (self.table_names, self.column_names,
             self.column_types, self.foreign_primary_keys, self.table_rows) = self._get_sqlite_schema(db_path_or_json)
        elif db_path_or_json.endswith(".json"):
            with open(db_path_or_json, "r") as f:
                schema_info = json.load(f)
                self.table_names = schema_info["tables_names_original"] if "tables_names_original" in schema_info \
                    else schema_info["table_names"]
                self.column_names = schema_info["column_names_original"] if "column_names_original" in schema_info \
                    else schema_info["column_names"]
                self.column_types = schema_info["column_types"]
                self.foreign_primary_keys = schema_info["foreign_keys"]
            self.table_rows = None
        else:
            raise ValueError

    def schema_as_dict(self) -> dict:
        return {
            "table_names": self.table_names,
            "column_names": self.column_names,
            "column_types": self.column_types,
            "foreign_primary_keys": self.foreign_primary_keys
        }

    @staticmethod
    def _get_sqlite_schema(sqlite_path: str) -> tuple:
        """
        Returns the database schema of a .sqlite file
        Args:
        sqlite_path (str): database path

        Returns (tuple): Returns a tuple with
                        'table_names': list[str],
                        'column_names': list[tuple(<table_index>, <column_name>)],
                        'column_types': list[str],
                        'foreign_primary_keys': list[tuple(<foreign_key_column_index, primary_key_column_index>)]}
        """

        conn = sqlite3.connect(sqlite_path)
        conn.execute('pragma foreign_keys=ON')

        # Get all the tables
        table_names = list(
            map(lambda row: row[0], conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()))

        # Initialize list of columns
        column_names = [(-1, '*')]
        column_types = [""]

        foreign_primary_keys_names = []
        tables_rows = []
        # For each table
        for t_i, table_name in enumerate(table_names):

            # Get the foreign keys of the table
            table_fks_info = conn.execute("PRAGMA foreign_key_list('{}') ".format(table_name)).fetchall()
            for _, _, pk_table, fk_column, pk_column, _, _, _ in table_fks_info:
                foreign_primary_keys_names.append(((table_name, fk_column, pk_table, pk_column)))

            # Get the table's columns names and types
            table_columns_info = list(map(lambda row: (row[1], row[2]),
                                          conn.execute("PRAGMA table_info('{}') ".format(table_name)).fetchall()))
            # For every column
            for column_name, column_type in table_columns_info:
                column_names.append((t_i, column_name))
                column_types.append(column_type)

            tables_rows.append(conn.execute(f"select count(*) from {table_name}").fetchall()[0][0])

        # Convert foreign-primary keys names to column indexes
        foreign_primary_keys = []
        for fk_table, fk_column, pk_table, pk_column in foreign_primary_keys_names:
            foreign_primary_keys.append((column_names.index((table_names.index(fk_table), fk_column)),
                                         column_names.index((table_names.index(pk_table), pk_column))))

        return table_names, column_names, column_types, foreign_primary_keys, tables_rows

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

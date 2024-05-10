import sqlite3
from loguru import logger

from DatasetAnalysisTools.DatabaseInfo.databases.databaseABC import DatabaseABC


class SqliteDatabase(DatabaseABC):

    def __init__(self, sqlite_path: str):

        self.sqlite_path = sqlite_path

    def get_schema(self) -> tuple:
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

        conn = sqlite3.connect(self.sqlite_path)
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

            tables_rows.append(conn.execute(f"select count(*) from '{table_name}'").fetchall()[0][0])

        # Convert foreign-primary keys names to column indexes
        foreign_primary_keys = []

        for fk_table, fk_column, pk_table, pk_column in foreign_primary_keys_names:
            try:
                foreign_primary_keys.append((column_names.index((table_names.index(fk_table), fk_column)),
                                             column_names.index((table_names.index(pk_table), pk_column))))
            except Exception as e:
                logger.warning(f"{e}. The fp relation will not be considered!")
                pass

        return table_names, column_names, column_types, foreign_primary_keys, tables_rows

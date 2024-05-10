import json

from DatasetAnalysisTools.DatabaseInfo.databases.databaseABC import DatabaseABC


class JsonDatabase(DatabaseABC):

    def __init__(self, json_path: str, db_id: str = None):

        self.json_path = json_path
        self.db_id = db_id

    def get_schema(self) -> tuple:
        with open(self.json_path, "r") as f:
            file_contents = json.load(f)
            # If there are information for more than one database in the file
            if len(file_contents) > 1:
                if self.db_id is None:
                    raise ValueError(
                        "The .json database file contains information for more than one databases and db_id is not "
                        "given!")
                else:
                    for db_content in file_contents:
                        if db_content["db_id"] == self.db_id:
                            schema_info = db_content
            elif len(file_contents) == 1:
                schema_info = file_contents[0]
            else:  # if there is a dict with the database info
                schema_info = file_contents

            table_names = schema_info["table_names_original"] if "table_names_original" in schema_info \
                else schema_info["table_names"]
            column_names = schema_info["column_names_original"] if "column_names_original" in schema_info \
                else schema_info["column_names"]
            column_types = schema_info["column_types"]
            foreign_primary_keys = schema_info["foreign_keys"]

        return table_names, column_names, column_types, foreign_primary_keys, None

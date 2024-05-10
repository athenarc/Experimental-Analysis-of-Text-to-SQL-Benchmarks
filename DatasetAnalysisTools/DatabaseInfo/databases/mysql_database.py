import pandas as pd
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
import os

from DatasetAnalysisTools.DatabaseInfo.databases.databaseABC import DatabaseABC

load_dotenv()


class MySQLDatabase(DatabaseABC):
    def __init__(self, db_path_or_name: str):

        self.engine = create_engine(f"mysql+pymysql://{os.getenv('username')}:{os.getenv('password')}@"
                                    f"{os.getenv('hostname')}:{os.getenv('port')}/{db_path_or_name}")

    def execute(self, query: str) -> pd.DataFrame:
        with self.engine.begin() as conn:
            df = pd.read_sql(text(query), con=conn)
        conn.close()
        self.engine.dispose()

        return df

    def get_schema(self) -> tuple:
        inspector = inspect(self.engine)

        table_names = inspector.get_table_names()

        column_names = [(-1, "*")]
        column_types = [""]
        table_rows = []

        for i, table_name in enumerate(table_names):
            for column in inspector.get_columns(table_name=table_name):
                column_names.append((i, column["name"]))
                column_types.append(column["type"])

            table_rows.append(self.execute(f"select count(*) as 'rows' from {table_name}")["rows"][0])

        # TODO add fp keys
        return table_names, column_names, column_types, [], table_rows


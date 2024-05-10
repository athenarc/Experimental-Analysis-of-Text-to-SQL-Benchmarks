from loguru import logger
import signal
from sqlalchemy.exc import ProgrammingError, OperationalError

from DatasetAnalysisTools.DatabaseInfo.databases.mysql_database import MySQLDatabase


def timeout_handler(signum, frame):
    raise TimeoutError()


def mysql_exec_evaluator(db_name: str, gold: str, predicted: str, timeout: int = 120000) -> dict:
    # Connect to database
    db = MySQLDatabase(db_name)

    try:
        result1 = db.execute(f"SELECT /*+ MAX_EXECUTION_TIME({timeout}) */ " + gold[len("select "):])
    except Exception as e:
        logger.warning(f"There was an error while executing the gold query {gold}! The execution accuracy will be set "
                       f"to 0.")
        raise SyntaxError(e)

    try:
        result2 = db.execute(f"SELECT /*+ MAX_EXECUTION_TIME({timeout}) */ " + predicted[len("select "):])

        return {"exec": (1 if result1.compare(result2).empty else 0) \
            if result1.shape[0] == result2.shape[0] and list(result1.columns) == list(result2.columns) else 0}
    except Exception as e:
        logger.warning(f"There was an error while executing the predicted query {predicted}! The execution accuracy "
                       f"will be set to 0.")
        raise SyntaxError(e)

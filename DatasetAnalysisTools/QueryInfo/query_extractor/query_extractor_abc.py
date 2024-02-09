from abc import abstractmethod

from DatasetAnalysisTools.QueryInfo.query_info import QueryInfo
from DatasetAnalysisTools.DatabaseInfo.database_info import DatabaseInfo


class QueryExtractor:

    @abstractmethod
    def extract(self, query: str, db_info: DatabaseInfo = None) -> QueryInfo:
        pass

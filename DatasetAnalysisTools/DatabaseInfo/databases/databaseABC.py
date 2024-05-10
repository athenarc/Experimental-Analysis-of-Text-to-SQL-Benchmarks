from abc import ABC, abstractmethod


class DatabaseABC(ABC):

    @abstractmethod
    def __init__(self, db_path_or_name, **kwargs):
        pass

    @abstractmethod
    def get_schema(self) -> tuple:
        pass

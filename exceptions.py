
class NotSupportedTask(Exception):
    """Raised when the task is not supported for a particular dataset"""
    def __init__(self):
        super().__init__("The requested task is not supported in this dataset!")


class NotSupportedDataset(Exception):
    """Raised when a dataset is not supported"""

    def __init__(self):
        super().__init__("Unknown dataset!")


class NotFoundDataset(Exception):
    """Raised when a dataset is not found in the given path"""
    def __init__(self):
        super().__init__("The requested dataset does not exist in the provided location!")


class MissingRequiredInfo(Exception):
    """ Raised when a required parameter or variable has not been defined"""
    def __init__(self, message):
        super().__init__(message)


class SQLParserException(Exception):
    """Raised when there is an error in parsing a SQL query with a selected parser"""
    def __init__(self, message):
        super().__init__(message)

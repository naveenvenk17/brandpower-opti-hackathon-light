
class DataNotFoundError(Exception):
    """Exception raised when data is not found."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details

class DataValidationError(Exception):
    """Exception raised for errors in data validation."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


class WellException(Exception):
    "All well exceptions"

class DryWellException(WellException):
    "Raised when an operation requires a well to have data, but the well doesn't."

class WellCacheException(WellException):
    "Raised when Caching fails"
from __future__ import annotations


class CollectiveIllegalStateError(RuntimeError):
    pass


class CollectiveConsistencyError(RuntimeError):
    pass


class CollectiveClosedError(CollectiveConsistencyError):
    pass


class DataUnavailable(Exception):
    """Raised when data is not available in the buffer"""

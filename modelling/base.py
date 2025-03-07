from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract BaseModel that defines the interface for all machine learning models.
    Every model must implement train(), predict(), and print_results() methods.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self, X_test) -> None:
        pass

    @abstractmethod
    def print_results(self, data) -> None:
        pass

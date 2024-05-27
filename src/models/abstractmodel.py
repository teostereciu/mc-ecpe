from abc import ABC, abstractmethod
import tensorflow as tf

class AbstractModel(ABC):
    """ Model abstract class. """
    def __init__(self, hyperpars):
        super().__init__()
        self.hyperpars = hyperpars

    @abstractmethod
    def get_model(self):
        pass

    
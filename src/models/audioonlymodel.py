from src.models.abstractmodel import AbstractModel


class AudioOnlyModel(AbstractModel):
    def __init__(self, hyperpars):
        super().__init__()

    def call(self, inputs):
        pass
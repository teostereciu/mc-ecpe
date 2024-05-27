from src.models.abstractmodel import AbstractModel
import tensorflow as tf 
#import tensorflow_hub as hub
import os

class TextAudioFusionModel(AbstractModel):
    def __init__(self, hyperpars):
        super().__init__(hyperpars)

    def get_model(self):
        pass


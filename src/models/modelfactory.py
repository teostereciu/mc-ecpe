from src.models import audioonlymodel
from src.models.abstractmodel import AbstractModel
from src.models.textaudiofusionmodel import TextAudioFusionModel
from src.models.textonlymodel import TextOnlyModel


class ModelFactory:
    """
    Naive factory method implementation for
    model creation. Can create a text-only, audio-only, 
    or text-and-audio model.
    """

    @staticmethod
    def create_model(model_type: str, hyperpars: dict[str, float]) -> AbstractModel:
        """
        Factory method for model creation.
        :param agent_type: a string key corresponding to the model.
        :param hyperpars: hyperparameter values
        :return: an object of type AbstractModel.
        """
        

        if model_type == "TEXT-ONLY":
            return TextOnlyModel(hyperpars)
        elif model_type == "AUDIO-ONLY":
            return audioonlymodel(hyperpars)
        elif model_type == "TEXT-AND-AUDIO":
            return TextAudioFusionModel(hyperpars)

        raise ValueError("Invalid model type")
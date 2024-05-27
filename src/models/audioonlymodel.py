import tensorflow as tf
from src.models.abstractmodel import AbstractModel


class AudioOnlyModel(AbstractModel):
    sequence_length = 100
    num_features = 18
    def __init__(self, hyperpars):
        super().__init__(hyperpars)
        self.model = self.get_model()
        self.sequence_length = 100
        self.num_features = 18
    
    

    def get_model(self):
        #inputs = [tf.keras.layers.Input(self.sequence_length,) for _ in range(self.num_features)]
        
        input_layer = tf.keras.layers.Input(shape=(self.sequence_length, self.num_features))

        #stacked = tf.keras.layers.Concatenate(axis=-1)(inputs)
        
        lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(input_layer)
  
        #x = tf.keras.laters.Dense(512, activation='relu')(lstm_1)
        x = tf.keras.layers.Dense(256, activation='relu')(lstm_1)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        outputs = tf.keras.layers.Dense(7, activation='softmax')

        return tf.keras.Model(input_layer, outputs)

    def train_model(self, X_train, y_train, X_val, y_val):
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val))
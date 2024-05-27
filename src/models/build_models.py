import tensorflow as tf

def build_audio_only_model(sequence_length=100, num_features=18):
    input_layer = tf.keras.layers.Input(shape=(sequence_length, num_features))
    lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(input_layer)
  
    #x = tf.keras.laters.Dense(512, activation='relu')(lstm_1)
    x = tf.keras.layers.Dense(256, activation='relu')(lstm_1)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)

    model = tf.keras.Model(input_layer, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def build_model(type):
    if type == 'AUDIO-ONLY':
        return build_audio_only_model()
    return None
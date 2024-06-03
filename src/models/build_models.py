import tensorflow as tf

#from src.models.uncertainty import RBFClassifier, add_gradient_penalty, add_l2_regularization, duq_training_loop

def build_audio_only_model(hp, num_features=988):
    input_layer = tf.keras.layers.Input(shape=(num_features,))
    x = tf.keras.layers.Dense(hp.choice('units', [16, 32, 64, 128]), activation='relu')(input_layer)
    
    hp_dense = hp.choice('units', [1, 2, 3])
    for i in hp_dense:
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(hp.choice('units', [16, 32, 64, 128]), activation='relu')(x)

    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)

    '''model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        RBFClassifier(7, 0.1, centroid_dims=2, trainable_centroids=True)
    ])'''

    model = tf.keras.Model(input_layer, outputs)
    optimizer = tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-7, 1e-2, sampling='log'))
    #optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    #add_gradient_penalty(model, lambda_coeff=0.5)
    #add_l2_regularization(model)

    return model


#from transformers import TFAutoModelForSequenceClassification, create_optimizer

def build_text_only_model(num_features=1024):
    input_layer = tf.keras.layers.Input(shape=(num_features,))
    x = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    
    model = tf.keras.Model(input_layer, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, y, **kwargs):
        attn, attention_scores = self.mha(
            query=x, value=y,
            return_attention_scores=True
        )

        self.last_attention_scores = attention_scores

        x = self.add([x, attn])
        return self.layernorm(x)


def build_text_audio_model(num_text_features=1024, num_audio_features=988, num_classes=7, num_heads=8, key_dim=64):

    text_input = tf.keras.layers.Input(shape=(num_text_features,))
    audio_input = tf.keras.layers.Input(shape=(num_audio_features,))

    audio_reshaped = tf.keras.layers.Reshape((1, -1))(audio_input)
    text_reshaped = tf.keras.layers.Reshape((1, -1))(text_input)

    text_to_audio_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(x=text_reshaped, y=audio_reshaped)

    audio_to_text_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(x=audio_reshaped, y=text_reshaped)

    concatenated_attention = tf.keras.layers.Concatenate()([text_to_audio_attention, audio_to_text_attention])
    _, _, num_features = concatenated_attention.shape
    print(concatenated_attention.shape)
    reshaped_attn = tf.keras.layers.Reshape((-1,))(concatenated_attention)

    x = tf.keras.layers.Dense(512, activation='relu')(reshaped_attn)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x) 

    '''x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x) 
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)  
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x) ''' 
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[text_input, audio_input], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def build_model(type):
    if type == 'AUDIO-ONLY':
        return build_audio_only_model()
    if type == 'TEXT-ONLY':
        return build_text_only_model()
    if type == 'TEXT-AND-AUDIO':
        return build_text_audio_model()
    return None

import tensorflow as tf

#from src.models.uncertainty import RBFClassifier, add_gradient_penalty, add_l2_regularization, duq_training_loop

def build_audio_only_model(sequence_length=100, num_features=988):
    input_layer = tf.keras.layers.Input(shape=(num_features,))
    x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)


    #duq = RBFClassifier(7, 0.1, centroid_dims=2, trainable_centroids=True)(x)

    #x = tf.keras.layers.Dense(32, activation='relu')(x)
    #x = tf.keras.layers.Dense(32, activation='relu')(x)
    #x = tf.keras.layers.Dense(64, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.2)(x)

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    #optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    #add_gradient_penalty(model, lambda_coeff=0.5)
    #add_l2_regularization(model)

    return model


#from transformers import TFAutoModelForSequenceClassification, create_optimizer

def build_text_only_model(num_features=768):
    input_layer = tf.keras.layers.Input(shape=(num_features,))
    x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    #x = tf.keras.layers.Dropout(0.2)(x)
    #x = tf.keras.layers.Dense(32, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    #x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    
    model = tf.keras.Model(input_layer, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model(type):
    if type == 'AUDIO-ONLY':
        return build_audio_only_model()
    if type == 'TEXT-ONLY':
        return build_text_only_model()
    return None

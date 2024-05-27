import numpy as np
from src.models.abstractmodel import AbstractModel


#import tensorflow_hub as hub
import tensorflow as tf 
#import tensorflow_text
import os

from tensorflow.python.framework.ops import disable_eager_execution



class TextOnlyModel(AbstractModel):
    #os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub_modules'
    #hub_link = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
    #hub_link = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-24-h-1024-a-16/4"
    hub_preprocess = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"
    hub_bert = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-24-h-1024-a-16/4"

    def __init__(self, hyperpars, load = False, load_path = None, num_classes=7):
        super().__init__(hyperpars)

        if load:
            self.model = self.load_model(load_path)
        else:
            self.preprocessor, self.model = self.get_model()
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def get_model(self):
        print('ok0')
        return self.make_bert_preprocessor(), self.make_bert_classifier()

    def get_bert_layer(self):
        return self.model.get_layer(name='bert')
    
    def make_bert_preprocessor(self, seq_length=40):
        '''input_segments = [
              tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
              ]
        print('ok')
        # Tokenize the text to word pieces.
        bert_preprocess = hub.load(self.hub_preprocess)
        tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
        segments = [tokenizer(s) for s in input_segments]
        print('ok1')
        # Optional: Trim segments in a smart way to fit seq_length.
        truncated_segments = segments

        # Pack inputs. The details (start/end token ids, dict of output tensors)
        # are model-dependent, so this gets loaded from the SavedModel.
        packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                                  arguments=dict(seq_length=seq_length),
                                  name='packer')
        print('ok2')
        model_inputs = packer(truncated_segments)
        print('ok3')
        return tf.keras.Model(input_segments, model_inputs)'''

    def make_bert_classifier(self, seq_length=40, num_classes=7):
        
        '''input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="input_type_ids")'''
        
        '''encoder_inputs = dict(
            input_word_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
            input_mask=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
            input_type_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        )
        
        encoder = hub.KerasLayer(
            self.hub_bert,
            trainable=False,
            name='bert')
        #bert_layer = hub.KerasLayer(self.hub_link, name="bert", trainable=False)
        encoder_output = encoder(encoder_inputs)
        pooled_output = encoder_output['pooled_output']
        #pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

        dense_layer1 = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
        #dense_layer2 = tf.keras.layers.Dense(512, activation='relu')(dense_layer1)
        #dense_layer3 = tf.keras.layers.Dense(128, activation='relu')(dense_layer2)
        output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dense_layer1)
        
        #return tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output_layer)
        return tf.keras.Model(inputs=encoder_inputs, outputs=output_layer)'''

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        '''return self.model.fit(x=X_train, 
                        y=y_train, 
                        validation_data=(X_val, y_val),
                        batch_size=32,
                        epochs=1)'''
    
    def save_model(self, path):
        bert_path = path / 'bert'
        model_path = path / 'text-model.h5'
        weights_path = path / 'text-model-weights.h5'
        self.get_bert_layer().export(bert_path)
        self.model.save(model_path)
        self.model.save_weights(weights_path)

    def load_model(self, path):
        return tf.keras.models.load_model(path)

    def load_model_weights(self, path):
        self.model.load_weights(path)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

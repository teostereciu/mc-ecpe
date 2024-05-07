import numpy as np
from src.models.abstractmodel import AbstractModel


import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer     
import os


class TextOnlyModel(AbstractModel):
    os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub_modules'
    hub_link = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
    
    def __init__(self, hyperpars, num_classes=7):
        super().__init__()
        self.hyperpars = hyperpars
        max_seq_length = 50
        #max_seq_length = hyperpars['max_seq_len']

        self.input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
        self.input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
        self.segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
        self.bert_layer = hub.KerasLayer(self.hub_link, trainable=True)

        self.dense_layer1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_layer2 = tf.keras.layers.Dense(num_classes, activation='softmax')


    def call(self, inputs, training=False):
        input_word_ids = inputs['input_word_ids']
        input_mask = inputs['input_mask']
        segment_ids = inputs['segment_ids']

        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
    
        x = self.dense_layer1(pooled_output) 
        output = self.dense_layer2(x)

        return output

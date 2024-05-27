import click
#import json
import logging
#import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
#from bert.tokenization import FullTokenizer   

#from src.features.text_preprocessing import get_ids, get_masks, get_segments
from src.features.audio_preprocessing import get_audio_features
from src.features.text_preprocessing import get_text_features
from src.models.build_models import build_model
from src.models.modelfactory import ModelFactory
#from dotenv import find_dotenv, load_dotenv
    
# need to do this for the bert preprocessing
#import sys
#from absl import flags

from src.visualization.visualize import plot_history


def encode_labels(text_labels):
    """ One-hot encodes emotion labels. """

    labels = ['neutral', 'joy', 'surprise', 'anger', 'fear', 'disgust', 'sadness']
    num_classes = len(labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for index, label in enumerate(labels)}

    idx_labels = [label_to_index[label] for label in text_labels]
    onehot_labels = tf.one_hot(idx_labels, len(labels)).numpy()
    return onehot_labels

def get_labels(df):
    train_labels = df[df['split'] == 'train']['emotion']
    val_labels = df[df['split'] == 'val']['emotion']
    test_labels = df[df['split'] == 'test']['emotion']
    return encode_labels(train_labels), encode_labels(val_labels), encode_labels(test_labels)

@click.command()
@click.option('--model_type', required=True, type=str)
def main(model_type):
    """ Trains a model.
    """
    logger = logging.getLogger(__name__)
    #hyperpars = get_hyperpars()

    max_seq_length = 40


    logger.info('Building model')
    model = build_model(model_type)
    logger.info('Done building model')
    fname = processed_data_dir / 'processed_data.csv'

    logger.info('Getting data')
    df = pd.read_csv(fname, dtype={'id':str})
    rename_split = {'train_wav': 'train', 'dev_wav': 'val', 'test_wav': 'test'}
    df['split'] = df['split'].map(lambda x: rename_split[x])
    
    logger.info('Text')
    X_train_text, X_val_text, X_test_text = get_text_features(df)
    logger.info('Audio')
    X_train_audio, X_val_audio, X_test_audio = get_audio_features(df)
    logger.info('Labels')
    y_train, y_val, y_test = get_labels(df)

    logger.info('Done getting data')

    logger.info('Training model')
    history = model.fit(X_train_audio, y_train, validation_data=(X_val_audio, y_val), epochs=3, batch_size=64)
    logger.info('Done training')

    plot_history(history, 'loss')
    plot_history(history, 'accuracy')

    '''model_factory = ModelFactory()
    model_wrapper = model_factory.create_model(model_type, None)
    
    bert = None
    bert = model_wrapper.get_bert_layer()

    fname = processed_data_dir / 'processed_data.csv'
    df = pd.read_csv(fname, dtype={'id':str})
    rename_split = {'train_wav': 'train', 'dev_wav': 'val', 'test_wav': 'test'}
    df['split'] = df['split'].map(lambda x: rename_split[x])
    
    X_train_text, X_val_text, X_test_text = get_text_features(df)
    X_train_audio, X_val_audio, X_test_audio = get_audio_features(df)
    y_train, y_val, y_test = get_labels(df)

    print(X_train_audio)
    print(y_train)


    #X_train_bert = model_wrapper.preprocessor(X_train)
    #X_val_bert = model_wrapper.preprocessor(X_val)

    #history = model_wrapper.train_model(X_train_bert, y_train, X_val_bert, y_val)
    history = model_wrapper.train_model(X_train_audio, y_train, X_val_audio, y_val)
    
    plot_history(history, 'loss')
    plot_history(history, 'accuracy')

    #model_wrapper.save_model(models_dir)'''


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    project_dir = Path(__file__).resolve().parents[2]
    #raw_data_dir = project_dir / 'data' / 'raw'
    interim_data_dir = project_dir / 'data' / 'interim'
    processed_data_dir = project_dir / 'data' / 'processed'
    models_dir = project_dir / 'models'
    #json_data_path = raw_data_dir / 'Subtask_2_train.json'
    #dotenv_path = os.path.join(project_dir, '.env')
    #load_dotenv(dotenv_path)
    

    main()

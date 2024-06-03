import click
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.features.audio_preprocessing import get_audio_features, do_oversample, do_scale, do_pca
from src.features.text_preprocessing import bert_encode_text, get_text_features
from src.models.build_models import build_model

from src.visualization.visualize import plot_history, plot_precision_recall


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
    return train_labels, val_labels, test_labels


def get_data(fname, model_type):
    logger = logging.getLogger(__name__)
    
    df = pd.read_csv(fname, dtype={'id':str})

    logger.info('Labels')
    y_train, y_val, y_test = get_labels(df)

    def get_audio(y_train, oversample=True, scale=True, yes_pca=False):
        logger.info('Audio')
        X_train_audio, X_val_audio, X_test_audio = get_audio_features(df)
        if oversample:
            logger.info('Oversampling')
            X_train_audio, y_train = do_oversample(X_train_audio, y_train)
        
        if scale:
            logger.info('Scaling')
            X_train_audio, scaler = do_scale(X_train_audio)
            X_val_audio = do_scale(X_val_audio, scaler)
            X_test_audio = do_scale(X_test_audio, scaler)
        
        if yes_pca:
            logger.info('PCA')
            X_train_audio, pca = do_pca(X_train_audio)
            X_val_audio = do_pca(X_val_audio, pca)
            X_test_audio = do_pca(X_test_audio, pca)

        return X_train_audio, X_val_audio, X_test_audio, y_train

    def get_text(y_train, from_file=True, oversample=True):
        logger.info('Text')
        if from_file:
            bert_train = np.load('/Users/teodorastereciu/Documents/bachelors-project/mc-ecpe/data/processed/bert_large_train.npy')
            bert_val = np.load('/Users/teodorastereciu/Documents/bachelors-project/mc-ecpe/data/processed/bert_large_val.npy')
            bert_test = np.load('/Users/teodorastereciu/Documents/bachelors-project/mc-ecpe/data/processed/bert_large_test.npy')
        else:
            X_train_text, X_val_text, X_test_text = get_text_features(df)
            bert_train, bert_val, bert_test = bert_encode_text(X_train_text, X_val_text, X_test_text)
            
        if oversample:
            logger.info('Oversampling')
            bert_train, y_train = do_oversample(bert_train, y_train, 'RANDOM')
        
        return bert_train, bert_val, bert_test, y_train
    
    def get_both():
        audio_train, audio_val, audio_test, _ = get_audio(y_train, oversample=False)
        bert_train, bert_val, bert_test, _ = get_text(y_train, oversample=False)
        return [bert_train, audio_train], [bert_val, audio_val], [bert_test, audio_test]
    
    if model_type == "AUDIO-ONLY":
        X_train, X_val, X_test, y_train = get_audio(y_train)
    elif model_type == "TEXT-ONLY":
        X_train, X_val, X_test, y_train = get_text(y_train)
    else:
        X_train, X_val, X_test = get_both()


    y_train_onehot = encode_labels(y_train)
    y_val_onehot = encode_labels(y_val)
    y_test_onehot = encode_labels(y_test)

    return X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot


@click.command()
@click.option('--model_type', required=True, type=str)
def main(model_type):
    """ Trains a model.
    """
    logger = logging.getLogger(__name__)
    #hyperpars = get_hyperpars()

    logger.info('Building model')
    model = build_model(model_type)
    logger.info('Done building model')

    print(model.summary())

    logger.info('Getting data')
    fname = processed_data_dir / 'processed_func_data.csv'
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(fname, model_type)
    logger.info('Done getting data')

    
    logger.info('Training model')
    #print(X_train.shape)
    #print(y_train.shape)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300, batch_size=32)
    logger.info('Done training')

    plot_history(history, 'loss')
    plot_history(history, 'accuracy')

    plot_precision_recall(y_val, model.predict(X_val))


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

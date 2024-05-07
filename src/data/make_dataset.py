# -*- coding: utf-8 -*-
#import click
import json
import logging
import os
from pathlib import Path

import opensmile
from dotenv import find_dotenv, load_dotenv


def get_data_from_json(json_convo):
    """ Extracts text, emotion, file name, and speaker labels from
        a json conversation.
    """

    lines = []
    emotions = []
    filenames = []
    speakers = []
    for line_idx in range(len(json_convo)):
        lines += [json_convo[line_idx]['text']]
        emotions += [json_convo[line_idx]['emotion']]
        filenames += [json_convo[line_idx]['video_name'][:-4]]
        speakers += [json_convo[line_idx]['speaker']]
    return lines, emotions, filenames, speakers

def extract_audio_features_from_wav(filename, split, smile):
    """ Extracts smile audio features from a wav file. """
    filepath = interim_data_dir / split / filename

    if not os.path.isfile(filepath):
        return None # split is unknown so we try everywhere, if file not here skip

    result_df = smile.process_file(filepath)
    features = result_df.to_numpy()
    return features


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building dataset')
    logger.info('loading json data')
    text_data = json.loads(json_data_path.read_text())
    convos = []
    all_lines = []
    all_emotions = []
    all_filenames = []
    all_speakers = []
    for convo_idx in range(len(text_data)):
        lines, emotions, filenames, speakers = get_data_from_json(text_data[convo_idx]['conversation'])
        convos += [lines]
        all_lines += lines
        all_emotions += emotions
        all_filenames += filenames
        all_speakers += speakers

    data_dict = {'filename': all_filenames, 'conversation' : convos, 'emotion': all_emotions, 'speaker': all_speakers}
    
    logger.info('done loading json data')
     # rn assume wav have been extracted from mp4 to ../interim
    logger.info('extracting smile features from wav')
    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    all_audio_features = []

    for fname in data_dict['filename']:
        features = extract_audio_features_from_wav(fname, 'train', smile)
        if features is None:
            features = extract_audio_features_from_wav(fname, 'dev', smile) 
        if features is None:
            features = extract_audio_features_from_wav(fname, 'test', smile) 
        all_audio_features += features
    
    logger.info('done extracting smile features from wav')
   




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / 'data' / 'raw'
    interim_data_dir = project_dir / 'data' / 'interim'
    processed_data_dir = project_dir / 'data' / 'processed'
    json_data_path = raw_data_dir / 'Subtask_2_train.json'
    

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

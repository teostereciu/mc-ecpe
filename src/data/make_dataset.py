# -*- coding: utf-8 -*-
#import click
import json
import logging
import os
from pathlib import Path

import opensmile
import pandas as pd
#from dotenv import find_dotenv, load_dotenv


def get_data_from_json(json_convo, convo_idx):
    """ Extracts text, emotion, file name, and speaker labels from
        a json conversation.
    """

    lines = []
    emotions = []
    filenames = []
    speakers = []
    ids = []
    for line_idx in range(len(json_convo)):
        lines += [json_convo[line_idx]['text']]
        emotions += [json_convo[line_idx]['emotion']]
        filenames += [json_convo[line_idx]['video_name'][:-4]]
        speakers += [json_convo[line_idx]['speaker']]
        # label convo lines so we can still track which conversations they came from
        # and what position they have within the conversation        
        id = str(convo_idx) + str(line_idx)
        ids += [id]
    return lines, emotions, filenames, speakers, ids

def extract_audio_features_from_wav(filename, split, smile):
    """ Extracts smile audio features from a wav file. """
    filename += '.wav'
    filepath = interim_data_dir / split / filename

    if not os.path.isfile(filepath):
        return None # split is unknown so we try everywhere, if file not here skip

    result_df = smile.process_file(filepath)

    collapsed_series = result_df.T.apply(lambda x: x.tolist(), axis=1)
    collapsed_df = collapsed_series.to_frame().T
    
    return collapsed_df
    

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building dataset')
    logger.info('loading json data')
    text_data = json.loads(json_data_path.read_text())

    all_lines = []
    all_emotions = []
    all_filenames = []
    all_speakers = []
    all_ids = []
    
    for convo_idx in range(len(text_data)):
        lines, emotions, filenames, speakers, ids = get_data_from_json(text_data[convo_idx]['conversation'], convo_idx)
        all_lines += lines
        all_emotions += emotions
        all_filenames += filenames
        all_speakers += speakers
        all_ids += ids

    data_dict = {'filename': all_filenames, 'id': all_ids, 'conversation_line' : all_lines, 'emotion': all_emotions, 'speaker': all_speakers}
    df = pd.DataFrame.from_dict(data_dict).astype({'id':str})

    logger.info('done loading json data')
    # rn assume wav have been extracted from mp4 to ../interim
    logger.info(f'extracting smile features from wav files in {interim_data_dir}')
    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b, 
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    all_features = []

    for fname in data_dict['filename']:
        features = extract_audio_features_from_wav(fname, 'train_wav', smile)
        if features is None:
            features = extract_audio_features_from_wav(fname, 'dev_wav', smile) 
        if features is None:
            features = extract_audio_features_from_wav(fname, 'test_wav', smile) 
        
        all_features += [features]

    audio_features_df = pd.concat(all_features, axis=0)
    audio_features_df.reset_index(drop=True, inplace=True)

    logger.info('done extracting smile features from wav')
    logger.info('creating final dataset')

    concatenated_df = pd.concat([df, audio_features_df], axis=1)
    save_path = processed_data_dir / 'processed_dataset.csv'
    concatenated_df.to_csv(save_path, index=False)  

    logger.info(f'done creating final dataset. saved to: {save_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / 'data' / 'raw'
    interim_data_dir = project_dir / 'data' / 'interim'
    processed_data_dir = project_dir / 'data' / 'processed'
    json_data_path = raw_data_dir / 'Subtask_2_train_small.json'
    

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()

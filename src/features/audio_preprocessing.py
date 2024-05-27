import numpy as np


def uniform_length(str, max_len=1000):
    ls = list(eval(str))
    if len(ls) < max_len:
        to_append = [0]*(max_len - len(ls))
        ls += to_append
    ls = ls[:max_len]
    return repr(ls)

def add_feature(srs, split, features):
    for feature in srs.index:
        str = srs[feature]
        ls = list(eval(str))
        if feature in features[split].keys():
            features[split][feature] += [ls]
        else:
            features[split][feature] = [ls]
        
    return features

def get_audio_features(df):
    #rename_split = {'train_wav': 'train', 'dev_wav': 'val', 'test_wav': 'test'}
    #df['split'] = df['split'].map(lambda x: rename_split[x])
    
    ignore = ['filename', 'id', 'conversation_line', 'emotion', 'speaker', 'split']
    audio_features = df.drop(columns=ignore)
    
    same_len_audio_features = audio_features.map(lambda x: uniform_length(x))
    
    features = {'train':{}, 'val':{}, 'test':{}}

    for i in range(len(same_len_audio_features.index)):
        srs = same_len_audio_features.iloc[i]
        split = df['split'].iloc[i]
        features = add_feature(srs, split, features)

    X_train = np.array(list(features['train'].values()))
    X_val = np.array(list(features['val'].values()))
    X_test = np.array(list(features['test'].values()))
    
    return np.transpose(X_train, (1, 0, 2)), np.transpose(X_val, (1, 0, 2)), np.transpose(X_test, (1, 0, 2))
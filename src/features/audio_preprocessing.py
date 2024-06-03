import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    
    ignore = ['filename', 'id', 'conversation_line', 'emotion', 'speaker', 'split']
    X_train_audio = np.array(df[df['split'] == 'train'].drop(columns=ignore).values.tolist())
    X_val_audio = np.array(df[df['split'] == 'val'].drop(columns=ignore).values.tolist())
    X_test_audio = np.array(df[df['split'] == 'test'].drop(columns=ignore).values.tolist())

    return X_train_audio, X_val_audio, X_test_audio

def do_oversample(X, labels, type='SMOTE'):
    strategy = {'disgust':500, 'fear':500}
    if type=='SMOTE':
        os = SMOTE(sampling_strategy=strategy)
    else:
        os = RandomOverSampler(sampling_strategy=strategy)
    X, labels = os.fit_resample(X, labels)
    return X, labels

def do_scale(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        return scaled_X, scaler
    else:
        return scaler.transform(X)

def do_pca(X, pca=None, n_comp=500):
    if pca is None:
        pca = PCA(n_comp)
        pca_X = pca.fit_transform(X)
        return pca_X, pca
    else:
        return pca.transform(X)
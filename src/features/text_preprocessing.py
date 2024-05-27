import numpy as np


def get_text_features(df, bert=None):
    #fname = processed_data_dir / 'processed_dataset.csv'
    #df = pd.read_csv(fname, dtype={'id':str})
    #rename_split = {'train_wav': 'train', 'dev_wav': 'val', 'test_wav': 'test'}
    #df['split'] = df['split'].map(lambda x: rename_split[x])
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    #ids = train_df['id'].to_list()
    #speakers = df['speaker'].to_list()
    train_text = train_df['conversation_line'].to_list()
    val_text = val_df['conversation_line'].to_list()
    test_text = test_df['conversation_line'].to_list()
    
    #return X_train, y_train, X_val, y_val
    return np.array(train_text),np.array(val_text), np.array(test_text)

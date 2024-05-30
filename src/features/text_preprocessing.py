import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, TFBertModel

def get_text_features(df):
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    train_text = train_df['conversation_line'].to_list()
    val_text = val_df['conversation_line'].to_list()
    test_text = test_df['conversation_line'].to_list()
    
    return train_text, val_text, test_text
    #return np.array(train_text),np.array(val_text), np.array(test_text)

 
def prep_text(model, ds, data_collator, shuffle=False):
    tf_set = model.prepare_tf_dataset(
        ds,
        shuffle=shuffle,
        batch_size=16,
        collate_fn=data_collator,
    )
    return tf_set

def prep_split_text(model, ds):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    def preprocess_function(ds):
        return tokenizer(ds['text'], truncation=True)
    ds = ds.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    tf_train_set = prep_text(model, ds['train'], data_collator, shuffle=True)
    tf_val_set = prep_text(model, ds['val'], data_collator)
    tf_test_set = prep_text(model, ds['test'], data_collator)

    return tf_train_set, tf_val_set, tf_test_set


def bert_encode_text(X_train, X_val, X_test):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True)
    model = TFBertModel.from_pretrained("google-bert/bert-base-uncased")

    print('Tokenizer is fast?', tokenizer.is_fast)

    def tkn(text):
        print('*')
        return tokenizer(text, return_tensors="tf", padding='max_length', truncation=True, max_length=30)
    
    def encode(X):
        batch_size = 300
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
        pooled_outputs = []
        i = 0
        for batch in dataset:
            print('batch', i)
            #print(batch.numpy().tolist())
            texts = [str(text.numpy(), 'utf-8') for text in batch]
            inputs = tkn(texts)
            outputs = model(inputs)
            pooled_output = outputs['pooler_output']
            pooled_outputs.append(pooled_output)
            i += 1
        return tf.concat(pooled_outputs, axis=0)

    return encode(X_train), encode(X_val), encode(X_test)
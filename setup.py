import os, re, json, yaml, argparse
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




#NMT
def process_translation_data(data_obj, trg_lang, data_volumn=101100):    
    min_len = 10 
    max_len = 300
    max_diff = 50
    volumn_cnt = 0

    corpus, processed = [], []
    
    for elem in data_obj:
        temp_dict = dict()
        x, y = elem['en'].strip().lower(), elem[trg_lang].strip().lower()
        x_len, y_len = len(x), len(y)

        #Filtering Conditions
        min_condition = (x_len >= min_len) & (y_len >= min_len)
        max_condition = (x_len <= max_len) & (y_len <= max_len)
        dif_condition = abs(x_len - y_len) < max_diff

        if max_condition & min_condition & dif_condition:
            corpus.append(x)
            corpus.append(y)
            processed.append({'x': x, 'y':y})
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == data_volumn:
                break

    return processed, corpus




def train_tokenizer(lang_pair, corpus):
    os.makedirs(f"data/{lang_pair}", exist_ok=True)
    corpus_path = f'data/{lang_pair}/corpus.txt'
    with open(corpus_path, 'w') as f:
        f.write('\n'.join(corpus))
    
    assert os.path.exists(corpus_path)
    assert os.path.exists('config.yaml')

    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']
    
    vocab_size = vocab_config['vocab_size']
    if lang_pair == 'multi':
        vocab_size *= 2

    tokenizer = Tokenizer(BPE(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=[
            vocab_config['pad_token'], 
            vocab_config['unk_token'],
            vocab_config['bos_token'],
            vocab_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(f"data/{lang_pair}/tokenizer.json")



def save_data(lang_pair, data_obj):

    if lang_pair != 'multi':
        train, valid, test = data_obj[:-1100], data_obj[-1100:-100], data_obj[-100:]
        data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}
    else:
        train, valid = data_obj[:-3000], data_obj[-3000:]
        data_dict = {k:v for k, v in zip(['train', 'valid'], [train, valid])}

    for key, val in data_dict.items():
        with open(f'data/{lang_pair}/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{lang_pair}/{key}.json')



def main(task):
    #Data Processing
    ende_data = load_dataset('wmt14', 'de-en', split='train')['translation']
    ende_data, ende_corpus = process_translation_data(ende_data, 'de')

    encs_data = load_dataset('wmt14', 'cs-en', split='train')['translation']
    encs_data, encs_corpus = process_translation_data(encs_data, 'cs')

    enru_data = load_dataset('wmt14', 'ru-en', split='train')['translation']
    enru_data, enru_corpus = process_translation_data(enru_data, 'ru')        


    #For Multi Lingual Training Data
    multi_data = []
    for de_elem, cs_elem, ru_elem in zip(ende_data, encs_data, enru_data):
        multi_data.append({'x': de_elem['x'], 'y': de_elem['y']})
        multi_data.append({'x': cs_elem['x'], 'y': cs_elem['y']})
        multi_data.append({'x': ru_elem['x'], 'y': ru_elem['y']})

    multi_data = multi_data[:303000]
    multi_corpus = ende_corpus + encs_corpus + enru_corpus


    #Train Toeknizer
    train_tokenizer('ende', ende_corpus)
    train_tokenizer('encs', encs_corpus)
    train_tokenizer('enru', enru_corpus)
    train_tokenizer('multi', multi_corpus)

    #Save Data
    save_data('ende', ende_data)
    save_data('encs', encs_data)
    save_data('enru', enru_data)
    save_data('multi', multi_data)



if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)

    args = parser.parse_args()
    assert args.task in ['multi-source', 'multi-targe', 'multi-lingual']

    main(args.task)
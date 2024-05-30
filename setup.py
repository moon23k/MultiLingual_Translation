import os, re, json, yaml, zipfile, unicodedata
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




def get_zfiles():
    zip_files = []

    #extract zip file names
    for root, _, _ in os.walk('data'):
        root = unicodedata.normalize('NFC', root)
        
        if '라벨링' in root:
            for zip in os.listdir(root):
                if 'AIHUB' in zip:
                    zip_files.append(f"{root}/{zip}")
    return zip_files




def extract_data(zip_files, min_len=10, max_len=200, max_cnt=11020):
    corpus, extracted_data = [], []
    domain_cnt = dict()

    for data_file in zip_files:
        with zipfile.ZipFile(data_file, 'r') as zip_ref:
            for _file in  zip_ref.namelist():
                with zip_ref.open(_file) as target_file:
                    json_content = target_file.read().decode('utf-8')
                    json_data = json.loads(json_content)
                
                domain = keyword = re.search(r'/(\D+)_', _file).group(1).strip()
                
                if '조례문' in domain:
                    continue
                
                if domain not in domain_cnt:
                    domain_cnt[domain] = 0
                
                for elem in json_data['data']:
                    x = elem['source_sentence']
                    y = elem['target_sentence']

                    x_len = len(x)
                    y_len = len(y)
                    min_condition = (x_len >= min_len) & (y_len >= min_len)
                    max_condition = (x_len <= max_len) & (y_len <= max_len)

                    if not max_condition & min_condition:
                        continue

                    if domain_cnt.get(domain, 0) < max_cnt:
                        corpus.append(x)
                        corpus.append(y)                        
                        extracted_data.append({'domain': domain, 'x': x, 'y': y})
                        domain_cnt[domain] += 1                        
                    else:
                        break

    #Save Corpus
    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))    

    return extracted_data




def train_tokenizer():
    corpus_path = f'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        tok_config = yaml.load(f, Loader=yaml.FullLoader)['tokenizer']

    tokenizer = Tokenizer(BPE(unk_token=tok_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=tok_config['vocab_size'], 
        special_tokens=[
            tok_config['pad_token'], 
            tok_config['unk_token'],
            tok_config['bos_token'],
            tok_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save("data/tokenizer.json")




def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-5100], data_obj[-5100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')




def main():
    zip_files = get_zfiles()
    extracted_data = extract_data(zip_files)    
    train_tokenizer()
    save_data(extracted_data)
    return




if __name__ == '__main__':
    main()


    
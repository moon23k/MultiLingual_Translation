import os
from tqdm import tqdm



def save_data(f_name, obj):
    with open(f'daily/seq/{f_name}', 'w') as f:
        f.write('\n'.join(obj))



def process(dataset):
    src, trg = [], []

    for dial in tqdm(dataset):
        seq = dial.split("__eou__")[:-1]
        
        if len(seq) % 2 == 1:
            seq = seq[:-1]
        
        for idx, sent in enumerate(seq):
            if idx % 2 == 0:
                src.append(sent)
            else:
                trg.append(sent)


    src_train, src_valid, src_test = src[:-6000], src[-6000:-3000], src[-3000:]
    trg_train, trg_valid, trg_test = trg[:-6000], trg[-6000:-3000], trg[-3000:]

    save_data('train.src', src_train)
    save_data('valid.src', src_valid)
    save_data('test.src', src_test)

    save_data('train.trg', trg_train)
    save_data('valid.trg', trg_valid)
    save_data('test.trg', trg_test)



if __name__ == '__main__':
    with open("daily/seq/raw.txt", 'r', encoding='utf-8') as f:
        dataset = f.readlines()
    
    process(dataset)
    os.remove("daily/seq/raw.txt")
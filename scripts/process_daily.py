import os
from tqdm import tqdm



def save_data(f_name, obj):
    with open(f'daily/seq/{f_name}', 'w') as f:
        f.write('\n'.join(obj))



def process(dataset):
    src, trg = [], []

    for dial in tqdm(dataset):
        seq = dial.split("__eou__")[:-1]
        seq_len = len(seq)

        if seq_len < 2:
            continue

        elif seq_len == 2:
            src.append(seq[0])
            trg.append(seq[1])
            continue

        #Incase of seq_len is even
        elif seq_len % 2 == 0:
            src.extend(seq[0::2])
            trg.extend(seq[1::2])

            src.extend(seq[1:-1:2])
            trg.extend(seq[2::2])
        
        #Incase of seq_len is odds
        elif seq_len % 2 == 1:
            src.extend(seq[0:-1:2])
            trg.extend(seq[1::2])
            
            src.extend(seq[1::2])
            trg.extend(seq[2::2])   


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
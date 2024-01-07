## Multi-Lingual Translaor
&nbsp; In a typical Machine Translation task, models are primarily designed to learn and infer for a single language pair. However, when faced with the need to translate the same source language into various different target languages, training and operating separate models for each language pair can be inefficient. Therefore, in this repo, we aim to train Translation Models on diverse language pairs while simultaneously comparing their performance to models trained individually for each language pair. Our goal is to explore the capabilities of Multi-Linguality.

<br><br> 


## Results

| | En-De | En-Cs | En-Ru |
|---|:---:|:---:|:---:|
| En-De Model || - | - |
| En-Cs Model | - |  | - |
| En-Ru Model || - | - |
| Multi Lingual Model | - | - | - |
| Multi Lingual Large Model | - | - | - |

<br><br> 

## How to Use

Clone the repo in your env
```
git clone https://github.com/moon23k/NMT_MultiLingual
```

<br>

Setup Datasets and Tokenizer via "setup.py" 
```
python3 setup.py
```

<br>

Actual Process via run.py
```
python3 run.py -mode ['train', 'test', 'inference']
               -langpair ['ende', 'encs', 'enru', 'multi']
               -search ['greedy', 'beam'] (Optional)
```

<br><br> 

## Reference
* [**Attention Is All You Need**]()

<br> 

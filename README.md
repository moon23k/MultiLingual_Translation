## Multi-Lingual Translaor
&nbsp; In a typical Machine Translation task, models are primarily designed to learn and infer for a single language pair. However, when faced with the need to translate the same source language into various different target languages, training and operating separate models for each language pair can be inefficient. Therefore, in this repo, we aim to train Translation Models on diverse language pairs while simultaneously comparing their performance to models trained individually for each language pair. Our goal is to explore the capabilities of Multi-Linguality.

<br><br> 


## Experimental Setup

**Data**
> WMT14의 세 가지 언어쌍(En-De / En-Cs / En-Ru)을 사용합니다.
모든 언어쌍 데이터의 숫자는 동일하게 설정해서 사용. 다만 Vocab size의 경우 Multi Lingual에서만 두배로 적용

**Model**
> Standard Transformer 모델을 사용합니다.
다만 다양한 언어 쌍을 다루기 위한 Multi Lingual Model에는 사이즈가 상이한 두개의 모델을 실험군으로 사용합니다.

**Training**
> MLE Training Objective


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

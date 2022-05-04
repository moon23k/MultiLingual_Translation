# seqGAN

<br>

## Intro

This repo covers implementation of seqGAN with my own edits.

SeqGAN is an approach that applies GAN to NLP Task through Reinforcement Learning.

Main idea borrowed from seqGAN, but the ways of configuring models and Loss functions are different.

In the original paper, Policy Gradient was used by getting rewards from discriminator and Roll-outs.

But In my case, I rather used output of discriminator as penalty than rewards.

Although these two approaches look different, in fact the conclusion they are trying to draw is the same.



<br>

## Model Architecture

Just like in GAN, seqGAN also consists of generator and discriminator.

Only Difference between GAN and seqGAN lies in Adversarial Training Process.


### Generator

Generator generates a series of sequence in a way the model trained through its training session.

I chose Transformer Architecture with parameter sharings.

<br>


### Discriminator

Discriminator determine the difference between real and generated data

Discriminator is also composed of Transformer with parameter sharings. 



<br>

### seqGAN




## How to Use

Clone the repo in your env
```
git clone https://github.com/moon23k/seqGAN
```

<br>

Download and Process with following command
```
bash prepare_dataset.sh
```

<br>

Train Generator, Generate Samples, Train Discriminator, Train
```
bash run -a pretrain -m generator
bash run -a generate
bash run -a pretrain -m discriminator
bash run -a train
```


<br>

## Results
Chat Logs with Pretrained Generator
![image](https://user-images.githubusercontent.com/71929682/166625739-6c331847-1357-4d4e-9015-544a6d7e2afd.png)

Chat Logs with Generator trained with seqGAN process
![image](https://user-images.githubusercontent.com/71929682/166625346-7130696e-b4a0-4e0d-9527-9f8a5341c7c5.png)


Results above show more attractive and interestin conversation happens with Pre-trained Generator model.



## Reference

seqGAN
Policy Gradient
Transformer

<br>
<br>

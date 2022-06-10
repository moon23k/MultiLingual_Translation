# seqGAN


This repo covers implementation of seqGAN with my own edits. SeqGAN arcitecture apply GAN to NLG Task via Reinforcement Learning Technique. Main idea borrowed from seqGAN, but the ways of configuring models and Loss functions are somewhat different. In the original paper, Policy Gradient was used by getting rewards from discriminator and Roll-outs. But In my case, I rather used output of discriminator as penalty than rewards. Though these two approaches look different, the main goal to draw is the same.



<br>

## Architecture

Just like in GAN, seqGAN also consists of generator and discriminator.

The main purpose 

But seqGAN uses Policy Gradient for Adversarial Learning.


### Generator

Generator generates a series of sequence in a way the model trained through its training session.

I chose Transformer Architecture with parameter sharings.

<br>


### Discriminator

Discriminator determine the difference between real and generated data

Discriminator is also composed of Transformer with parameter sharings. 



<br>

### seqGAN

Generator's Object is to fool Discriminator.
But setting loss function with BCE based on discriminator's output, makes learning process hard to optimize.

In Adversarial Process, Loss function combines CrossEntropy and Discriminator's output.


<br>


## How to Use

Clone the repo in your env
```
git clone https://github.com/moon23k/seqGANso
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
**Chat Logs with Pretrained Generator**

![image](https://user-images.githubusercontent.com/71929682/166625739-6c331847-1357-4d4e-9015-544a6d7e2afd.png)

<br>

**Chat Logs with Generator trained with seqGAN process**

![image](https://user-images.githubusercontent.com/71929682/166625346-7130696e-b4a0-4e0d-9527-9f8a5341c7c5.png)

<br>

Results above show more attractive and interestin conversation happens with Pre-trained Generator model.



## Reference

seqGAN
Policy Gradient
Transformer

<br>
<br>

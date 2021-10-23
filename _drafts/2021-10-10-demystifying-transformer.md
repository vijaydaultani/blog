---
layout: post
comments: true
title:  "Demystifying Transformer!"
excerpt: "This post will help demystify transformer mechanism by presenting motivation behind the concepts and explaining the underlying mathematical equations piece by piece. Also, we will learn how does the transformer mechanism interact with conventional encoder-decoder architecture."
date:   2021-10-08 17:00:00
tags: transformer encoder-decoder rnn
---
> In this article, I will introduce topic of transformer which have taken NLP community on fire which was intially proposed by ([Vaswani et al.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)). Transfomer was proposed to eliminate the limitation at the very heart of RNN's, therefore sequential processing of symbols in the input sequence one at a time. Similar to my previous articles furthermore, in this article, I aim first to provide the motivation, followed by the relevant mathematical equations to help deepen the understanding of the topic.

{: class="table-of-content"}
* TOC
{:toc}

## Background
Transduction is described as mapping an input string to an output string. Traditionally, sequence transduction models were based on encoder-decoder architectures where both of them were modeled via a recurrent or convolutional neural network. Soon after the encoder-decoder architecture was proposed by sutskever et al. in 2014 concept of attention was proposed by Bahandau in year 2016. Attention was used to condition each output symbol on a dynamic context vector. We described conventional encoder-decoder and attention in my previous article on demystifying attention here. Please read that article to get detailed understanding of the topic.

## Why Self Attention

## Breif model description

We will come back on this at last once we have learned all the involved concepts

## Embedding

### I/P and O/P Embedding

### Positional Encoding

## Attention

### Q, K, V Information Retrieval

### Attention (Scaled dot attention)

### Multi Head Attention

### Masked Multi Head Attention

### Multi Head Attention b/w encoder and decoder

### Where is attention used in the model
We used that in 3 places mentioned above

## Pointwise Feed Forward N/W

## Limitations
Limitation 1:  Though parallelization was possible across training examples using the concept of batching but not within one single example. RNN generate a sequence of hidden states $$h_t$$ as a function of previous hidden state $$h_{t-1}$$ and the input for position t. This inherent sequential nature to process input symbol one at a time in an order prevents parallelization within training examples. 

**Attention** became an integral part of several sequence modelling and transduction model for various tasks. I wrote an article on Attention previously which you can find here. Though in the article we went through in detail about the bottleneck problem of conventional encoder-decoder. But to summarize it here when encoder squashed the complete input into a one single vector the information was lost and decoder had no way to realize the importance of different inputs for generating symbol at hand especially for long-range dependencies. And therefore attention mechanism was introduced to allow modelling on input dependencies irrespective of the input's symbol distance in input and output sequences.

**Self-Attention** also called intra-attention is an attention mechanism that relates different position of a single sequence in order to compute a representation of the sequence. Notice the similarity with the conventional encoder. Similar to encoder self-attention is computing a representation of the sequence but rather than processing the input tokens one at a time self-attention will process them in parallel. Second self-attention similar to attention will relate (find similarity similar to attention function) different positions of a single sequence.

**Transformer** was the first model which was completely based on self-attention in order to compute the representations of it's input and output without using any sequence aligned RNNs or CNNs. 

In the paper they used transformer for the machine translation task but transformer itself is a very general architecture that can be used for any sequence modelling and transductions tasks.


## Model Architecture
Ok now let's formally define the problem. Given input sequence of symbol representations $$(x_1,...,x_n)$$ encoder will map that to a sequence of continious representations $$z = (z_1,...,z_n)$$. Later given $$z$$, the decoder then gnerates generates an output sequence $$(y_1,...,y_n)$$ of symbols one element at a time. Similar to conventional encoder-decoder architecture the decoder was used in a autoregressive settings i.e. previously generated symbol was consumed as an input by the decoder to generate the next symbol.

## Understanding the concept of key, query and value

## 

### Encoder and Decoder Stacks


## Results
In this article my motivation is not to showcase and promote how effecient the transfomers are. One can refer to the paper for the details of the results but since we are on the topic let's breifly talk about the improvement on the task at hand i.e. machine translation.



---

Cited as:
```
@article{daultani2021transformer,
  title   = "Demystifying Transformer!",
  author  = "Daultani, Vijay",
  journal = "https://github.com/vijaydaultani/blog",
  year    = "2021",
  url     = "https://vijaydaultani.github.io/blog/2021/10/08/demystifying-transformer.html"
}
```

*If you notice mistakes and errors in this post, don't hesitate to contact me at [vijay dot daultani at gmail dot com], and I would be pleased to correct them!*

See you in the next post :D

## References

[1] Vaswani et al. ["Attention Is All You Need"](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) NeurIPS 2017.

[2] [Illustrated Guide to Transformers- Step by Step Explanation](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)

[3] [Transformers Explained Visually — Not Just How, but Why They Work So Well](https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3)
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

Though parallelization was possible across training examples using the concept of batching but not within one single example. RNN generate a sequence of hidden states $$h_t$$ as a function of previous hidden state $$h_{t-1}$$ and the input for position t. This inherent sequential nature to process input symbol one at a time in an order prevents parallelization within training examples. 

**Attention** became an integral part of several sequence modelling and transduction model for various tasks. I wrote an article on Attention previously which you can find here. Though in the article we went through in detail about the bottleneck problem of conventional encoder-decoder. But to summarize it here when encoder squashed the complete input into a one single vector the information was lost and decoder had no way to realize the importance of different inputs for generating symbol at hand especially for long-range dependencies. And therefore attention mechanism was introduced to allow modelling on input dependencies irrespective of the input's symbol distance in input and output sequences.

**Self-Attention** also called intra-attention is an attention mechanism that relates different position of a single sequence in order to compute a representation of the sequence. Notice the similarity with the conventional encoder. Similar to encoder self-attention is computing a representation of the sequence but rather than processing the input tokens one at a time self-attention will process them in parallel. Second self-attention similar to attention will relate (find similarity similar to attention function) different positions of a single sequence.

**Transformer** was the first model which was completely based on self-attention in order to compute the representations of it's input and output without using any sequence aligned RNNs or CNNs. 

In the paper they used transformer for the machine translation task but transformer itself is a very general architecture that can be used for any sequence modelling and transductions tasks.

## Why Self Attention
Self attention was used because of the three desiredata. 
1. Computation complexity per layer

2. Amount of computation that can be parallelized which is measured as minimum number of sequential operations 

3. Path length between long-range dependencies 

## Breif model description

To start with let's start with briefly explaining how does the model architecture for transformer looks like. As one should have noticed the whole architecture is splitted into two pieces encoder and decoder. In Transformer both encoder and decoder are organized into what is know as layers (also knows as blocks). Further each layer is composed for sublayers. Where on one hadn at encoder side each layer is composed of two sublayers, on the other hand each layer at decoder is composed of three sublayers. 


Also, in case of encoder embeddings summed with positional encoding is given as an input to the encoder. Whereas on the decoder side outputs which are shifted right by one position are given as input to the decoder. Similar to encoder, at decoder side embeddings of words in the output are summed with their relevant positional encoding are further fed in the decoder side.

![transformer_model_architecture]({{ '/assets/images/demystifying_transformer/transformer_model_architecutre.jpg
' | relative_url }})
{: style="text-align: center"}
*Fig. 1 The Transfomer - model architecture".*

Given we have understood the big picture let's dive deep into understanding all the associated concepts in detail in order to be able to make sense of the complete architecture later.

## Embedding

### I/P and O/P Embedding
As a common practise in sequence transduction models, the authors used learned embeddings to convert the input symbols and output symbols to vectors of dimension $$d_model$$. Also at the decoder side they learned the linear transformation and softmax function to be able to convert the decoders output to predict next-token probabilities.

### Positional Encoding
Since transformer process the sequence in parallel, essentially leading to no recurrence and no convolution operation. Still since we are processing the sequence which inherently have structure in it, therefore it is important to feed the seuqence information to the model. The authors do this by adding "positional embeddings" of the symbols to their relevant input embeddings and feed the summed up together version as an input to the bottom most layer of the input and decoder respectively. In the paper authors explored two different types of positional embeddings i.e. learned and fixed embedding. Apparently the authors found that both of the different types of embeddings works almost equal in performance. But the authors decided to go ahead with fixed embedding since it could easily scale on longer sequence of the input which was not seen in the training data.

$$
\begin{aligned}
P E_{(p o s, 2 i)} &=\sin \left(p o s / 10000^{2 i / d_{\text {modec }}}\right) \\
P E_{(\text {pos }, 2 i+1)} &=\cos \left(\text { pos } / 10000^{2 i / d_{\text {modcl }}}\right)
\end{aligned}
$$

## Attention

An attention function can be described as mapping a query and a key-value pairs to an output, where all query, key, value and output are vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed using a compatibility function of the query with the corresponding key.

In order to understand the concept of the attention it is required to few important concepts as below

### Q, K, V Information Retrieval
This concept comes from information retrireval. Think of how any information retrieval system, e.g. google search engine might be working. Whenever you insert any query, google search engine essentially matches the inserted query against serveral keys (documents title). And then the documents whose keys matches with the inputted query are then ranked to show the most relevant values (in this case actual documents) on the top of the search results. 

Similarly, we have concepts of query Q, key K, and value V here in transformer. 

If you remember the original content based attention concept introduced by Bahandau the attention score was computed by a simple FFNN. And it was computed based on the below equation

$$
e\left(s_{j-1}, h_{i}\right)=v_{a}^{\top} \tanh \left(W_{a} s_{j-1}+U_{a} h_{i}\right)
\label{eq:attention_score}
$$

Where $$h_i$$ is the hidden state from encoder and $$s_{j-1}$$ is the hidden state from decoder. The problem with the above formulation of the attention score is that if there are m input symbols and n output symbols the one need to compute the attention scores by running through the network $$m * n$$ number of times to find the compatibility between hidden states of encoder and decoder. This is expensive since we have to run through the FFNN network $$m* * n$$ number of times. 

Rather in case of transformer they first transform both the input and output symbold into a common space and then rather than relying on a computational heavy function such as FFNN they calculate the compatibility by simply using a dot product to calculate the compatibility.

The given description on the stackexcahnge of that the query is only from the decoder side and key is from the encoder side is not completely correct as we also have multi head attention both only within the encoder and also within the decoder side. 

Said that there are multiple types of attention score such as described below. Below show the most common form of attention. 

### Attention (Scaled dot attention)



### Multi Head Attention

### Masked Multi Head Attention

### Multi Head Attention b/w encoder and decoder

### Where is attention used in the model
We used that in 3 places mentioned above

## Pointwise Feed Forward N/W

## All concepts together

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
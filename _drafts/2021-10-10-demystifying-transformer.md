---
layout: post
comments: true
title:  "Demystifying Transformer!"
excerpt: "This post will help demystify transformer mechanism by presenting motivation behind the concepts and explaining the underlying mathematical equations piece by piece. Also, we will learn how does the transformer mechanism interact with conventional encoder-decoder architecture."
date:   2021-10-08 17:00:00
tags: transformer encoder-decoder rnn
---
> In this article, I will introduce the topic of Transformer which have taken NLP community on fire which was intially proposed by ([Vaswani et al.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)). Transfomer was proposed to eliminate the limitation at the very heart of Recurrent Neural Networks (RNN's), i.e. sequential processing of symbols in the input sequence one symbol at a time. Similar to my previous articles, in this article, I aim first to provide the motivation, followed by the explanation of the relevant mathematical equations to help deepen the understanding of the topic.

{: class="table-of-content"}
* TOC
{:toc}

## Background
Transduction is described as mapping an input string to an output string. Traditionally, sequence transduction models were based on encoder-decoder architectures where both of them were modeled via a RNN, CNN or any other kind of network. Such a setup of encoder-decoder architecture was first proposed by sutskever et al. in 2014. These proposed vanilla RNN's and later it's variants have had several limitations such as bottleneck problem, sequential processing, teacher forcing etc. To solve the bottleneck problem concept of soft or additive attention was proposed by Bahandau et al. in 2016. I have described conventional encoder-decoder and soft-attention in my previous article on [Demystifying attention](https://vijaydaultani.github.io/blog/2021/10/08/demystifying-attention.html). Feel free to read it if you are not familiar with the concept of attention and are interested in detailed understanding of the topic. One should notice that Transformers by ([Vaswani et al.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)) was proposed to solve another problem of RNN's i.e. sequential processing.

In conventional encoder-decoder setup of RNN's, one can utilize the concept of batching to achieve parallelization **across** input and output sequences. But parallelization **within one single** input and output sequence is not possible, since recurrent models typically factor computation along the symbol positions of the input and output sequences. This alignment between position and steps in computation time means RNN can generate a sequence of hidden states $$h_t$$ as a function of previous hidden state $$h_{t-1}$$ and the input for current position t. This inherent sequential nature to process input symbol one at a time in an order inhibits parallelization within training examples. 

![seq-to-seq-using-decoder]({{ '/assets/images/demystifying_attention/seq-to-seq-using-attention-example.drawio.png' | relative_url }})
{: style="text-align: center"}
*Fig. 1 Abstractive example of encoder-decoder architecture usage translating a English sentence "Better late then never" to Japanese "決して 遅く ない より 良い".*

Let's try to understand this alignment between position and computation steps visually in Fig.1. For now focus on the hidden state $$h_2$$ corresponding to second input symbol "late" will depend on previous hidden state $$h_1$$ and the current input symbol $$x_2$$ ("late"). 

Though attention mechanism solved the bottleneck problem i.e. allowing modeling to dependencies without regarding to their distance in the input or output sequence, however the core problem of sequential processing persisted as attention was used in conjuction with a RNN. Later with the goal of reducing the sequential computation several work were proposed for e.g. [ByteNet](https://research.google/pubs/pub48560/) and [ConvS2s](http://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf). These work mostly employed convolution operation as a building block to compute the hidden representations of input and output sequence in parallel but number of operation to relate two input and output symbols at arbitrary positions grew with distance, on contrary Transformers were able to keep this constant.

The authors employed the concept of self-attention that can relate different position of a single sequence in order to compute a representation of the sequence. This one previous line on self-attention will not make any sence for now but we will explain the concept later in detail in this article. In essence self-attention is very similar to conventional RNN and attention mechanism. Similar to conventional RNN since they both compute a latent representation of the input sequence, though the difference is where conventional RNN will process one symbol at a time, self-attention will process symbols in parallel. Similar to attention since they both will relate i.e. find similarity between two symbols, though the difference is attention usually was used to relate one output and one input symbol, self-attention will additionally relate two input symbols or two output symbols as well. You should notice that even though the raw ingredients for transformers i.e. concepts such as attention, self-attention etc existed previously but transformer was the first model that completely relied on self-attention to compute the representations of it's input and output without using any sequence aligned RNNs or CNNs. Even though in the original paper by Vaswani et al. they utilized transformer for the machine translation task but transformer in itself is a very general architecture that can be used for any sequence modelling and transductions tasks.


## Problem Definition
Ok now let's formally define the problem. Given input sequence of symbol representations $$(x_1,...,x_n)$$ encoder will map that to a sequence of continious representations $$z = (z_1,...,z_n)$$. Later given $$z$$, the decoder then gnerates generates an output sequence $$(y_1,...,y_n)$$ of symbols one element at a time. Similar to conventional encoder-decoder architecture the decoder was used in a autoregressive settings i.e. previously generated symbol was consumed as an input by the decoder to generate the next symbol.

## Why Self Attention
Self attention was used because of the three desiredata. 
1. Computation complexity per layer

2. Amount of computation that can be parallelized which is measured as minimum number of sequential operations 

3. Path length between long-range dependencies 

## Breif model description

To start with let's start with briefly explaining how does the model architecture for transformer looks like. As one should have noticed the whole architecture is splitted into two pieces encoder and decoder. In Transformer both encoder and decoder are organized into what is know as layers (also knows as blocks). Further each layer is composed for sublayers. Where on one hadn at encoder side each layer is composed of two sublayers, on the other hand each layer at decoder is composed of three sublayers. 


Also, in case of encoder embeddings summed with positional encoding is given as an input to the encoder. Whereas on the decoder side outputs which are shifted right by one position are given as input to the decoder. Similar to encoder, at decoder side embeddings of words in the output are summed with their relevant positional encoding are further fed in the decoder side.

<p align="center">
  <img width="500" src="/assets/images/demystifying_transformer/transformer_model_architecutre.jpg">
</p>
*Fig. 1 The Transfomer - model architecture.*


Given we have understood the big picture let's dive deep into understanding all the associated concepts in detail in order to be able to make sense of the complete architecture later.

## Embedding

### I/P and O/P Embedding
As a common practise in sequence transduction models, the authors used learned embeddings to convert the input symbols and output symbols to vectors of dimension $$d_model$$. Also at the decoder side they learned the linear transformation and softmax function to be able to convert the decoders output to predict next-token probabilities.

### Positional Encoding
Since transformer process the sequence in parallel, essentially leading to no recurrence and no convolution operation. Still since we are processing the sequence which inherently have structure in it, therefore it is important to feed the seuqence information to the model. The authors do this by adding "positional embeddings" of the symbols to their relevant input embeddings and feed the summed up together version as an input to the bottom most layer of the input and decoder respectively. In the paper authors explored two different types of positional embeddings i.e. learned and fixed embedding. Apparently the authors found that both of the different types of embeddings works almost equal in performance. But the authors decided to go ahead with fixed embedding since it could easily scale on longer sequence of the input which was not seen in the training data.

$$
P E_{(p o s, 2 i)}=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right)
\label{eq:positional_encoding_sin}
$$

$$
P E_{(\mathrm{pos}, 2 i+1)}=\cos \left(\text { pos } / 10000^{2 i / d_{\text {modcl }}}\right)
\label{eq:positional_encoding_cos}
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
Out of all different types of attention the authors of the paper decided to use a variation of dot product known as scaled dot product attention. They call it scaled because once the dot product is caculate it is divided by $$\sqrt{d_{k}}$$. Where input consist of query and key of dimension $$d_{k}$$. The reason we do scaling is beacuse when we multiply query and key together the output product can be large in magnitude and therefore have to divide it with $$\sqrt{d_{k}}$$ to scale. If you have question why do we divide by $$\sqrt{d_{k}}$$ remember both query and key are of same dimension i.e. made sense to involve some variation of $$d_{d}$$ but why specifically $$\sqrt{d_{k}}$$, the authors didn't provide any insight. Atlast we take softmax of the product between query and key in order to get relative weight distribution across keys. Finally values are weighted by the output of compatibility calculated between query and key. 


In practise, the attention function is computed in parallel on a batch of queries (Q), keys (K) and values (V) in parallel to make the computation effective using the fast matrix multiplication libraries. 

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
\label{eq:scaled_dot_attention}
$$

### Multi Head Attention
Rather than using single attention model with $$d_{model}$$ dimensional queries, keys and values, the authors found it was beneficial to linearly project the queries, keys and values h times with different learned linear  projecttions to $$d_{k}$$, $$d_{k}$$, and $$d_{v}$$ dimensions respectively. Multi head attentin allows model to jointly attend to information from different representative subspaces at different points. 

$$
\operatorname{MultiHead}(Q, K, V)=\text { Concat }\left(\text { head }_{1}, \ldots, \text { head }_{\mathrm{h}}\right) W^{O}
\label{eq:multi_head_attention}
$$

$$
\text { where head }_{\mathrm{i}}=\text { Attention }\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\label{eq:head_i}
$$

In \eqref{eq:multi_head_attention} the three matrix multiplications $$Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}$$ generate the linear transformation for queries, keys and values respectively.

In \eqref{eq:multi_head_attention} observe that the final multi head attention operates on complete Q, K and V which are of $$d_model$$ dimension. Notice the difference in $$Attention$$ function in \eqref{eq:attention_score} and \eqref{eq:multi_head_attention}. On one hand in \eqref{eq:attention_score} operates on input Q, K and V, rather in \eqref{eq:multi_head_attention} the input to the attention function is not directly the query, keys and values rather the linear transformation of queries, keys and values respectively. Essentially it means attention is not calculated on the unprocessed version of queries, keys and values each of $$d_{model}$$ dimension rather on linearly transformed version of dimension $$d_{k}, d_{k}, d_{v}$$ respectively. 

#### Masked Multi Head Attention
To the output from the product of query and key we apply a mask. In case of encoder self-attention this mask is used to mask out the padding values so that they don't participate in attention calculation. Similar to self-attention in encoder a mask is applied to the output of self-attention of decoder, but on contarary to encoder in case of decoder the motivation of applying the mask is to hide the future tokens so that they don't interfere with calculating attention scores for self-attention. 

### Where is attention used in the model
If you followed me till now you would have realized that we used attention in total of three places in the transformer architecture. 

First, within the encoder in the form of encoder self-attention. Well within self-attention layer all of the inputs i.e. queries, keys and values come from the same place, i.e. the output of the previous layer in the encoder.

Second, within the decoder in the form of decoder self-attention. Similar to self-attention in encoder's case all of the inputs i.e. queries, keys and values come from the same place, i.e. the output of the previous layer in the decoder.

Third, encoder-decoder attention layers. In this case different from self-attention case in encoder and decoder, the queries come from the previous decoder layer, and keys and values come from the output of the encoder. This connection between encoder and decoder allows the queries from decoder to attend all positions in the input sequence and works similar to  attention mechanism in case of a conventional encoder-decoder architecture.

## Pointwise Feed Forward N/W
The last sublayer (second for encoder and third for decoder) in each layer of both encoder and decoder is a feed forward sublayer which is a fully connected feed forward network. The network is applied on each position in the input separately and identically. The fully connected layer as described in \eqref{eq:fully_connected} consists of two linear transformations separated by non linearity activation (ReLU) in between.

$$
\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
\label{eq:fully_connected}
$$

One thing to notice is that though the FFN function is same across different positions, each layer uses different parameters.

## Revisiting Model Architecture 
Given we have already covered required relevant background about embeddings, attention, multi-head attention, and pointwise feed forward network now it's time to tie up all the concepts together in the form of the

### Encoder and Decoder Stacks
Encoder is composed of 6 identical layers stacked on top of each other. Each layer consist of two sublayers. First sublayer is multi-head self-attention and second sublayer is a simple position-wise fully connected feed-forward network. Each of the two sublayers have residual connections, followed by layer normalization.

Decoder similar to encoder is composed of six identical layers stacked on top of each other. Different from it's encoder counterpart each layer in decoder consist of three sublayers. First sublayer is a masked multi-head self-attention, followed by second sublayer of multi-head encoder-decoder attention, which is finally followed by position-wise fully connected feed-forward network. Similar to it's encoder counterpart each sublayer have a residual connection which is followed by the layer normalization.

Finally toward the end of the decoder we linearly transform the output of the decoder which is then passed through a softmax to generate the probabilities on set of output symbols.

## Results
In this article my motivation is not to showcase and promote how effecient the transfomers are. One can refer to the paper for the details of the results but since we are on the topic let's breifly talk about the improvement on the task at hand i.e. machine translation.

## Summary
Now that we have already understood all the required concepts, let's tie them together. To revise we started with the motivation for the embedding where we discussed both I/P, O/P and positional embeddings. Later we introduced the attention mechanism. Which was further broken down into the several sub concepts. To understand 




**Attention** became an integral part of several sequence modelling and transduction model for various tasks. I wrote an article on Attention previously which you can find here. Though in the article we went through in detail about the bottleneck problem of conventional encoder-decoder. But to summarize it here when encoder squashed the complete input into a one single vector the information was lost and decoder had no way to realize the importance of different inputs for generating symbol at hand especially for long-range dependencies. And therefore attention mechanism was introduced to allow modelling on input dependencies irrespective of the input's symbol distance in input and output sequences.

**Self-Attention** also called intra-attention is an attention mechanism that relates different position of a single sequence in order to compute a representation of the sequence. Notice the similarity with the conventional encoder. Similar to encoder self-attention is computing a representation of the sequence but rather than processing the input tokens one at a time self-attention will process them in parallel. Second self-attention similar to attention will relate (find similarity similar to attention function) different positions of a single sequence.

**Transformer** was the first model which was completely based on self-attention in order to compute the representations of it's input and output without using any sequence aligned RNNs or CNNs. 

In the paper they used transformer for the machine translation task but transformer itself is a very general architecture that can be used for any sequence modelling and transductions tasks.



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
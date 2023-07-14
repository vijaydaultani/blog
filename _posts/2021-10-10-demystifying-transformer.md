---
layout: post
comments: true
title:  "Demystifying Transformer!"
excerpt: "This post will help demystify transformer mechanism by presenting motivation behind the concepts and explaining the underlying mathematical equations piece by piece. Also, we will learn how does the transformer mechanism interact with conventional encoder-decoder architecture."
date:   2021-10-08 17:00:00
tags: transformer encoder-decoder rnn
---
> In this article, I will introduce the topic of the Transformer, which is one of the most widely used basic building blocks for most, if not all, state-of-the-art models for a wide variety of NLP tasks. The architecture of the Transformer was proposed by ([Vaswani et al.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)). It was introduced to eliminate a severe limitation of Recurrent Neural Networks (RNNs), i.e., sequential processing of symbols (i.e., one symbol at a time) in the input sequence. We will start with the motivation of different concepts introduced in the paper, and then we will delve deep into the relevant mathematical equations to help deepen the understanding of the topic.

{: class="table-of-content"}
* TOC
{:toc}

## Background
Before Transformer, dominant sequence transduction models were based on encoder-decoder architectures where both encoder and decoder were modeled via an RNN, CNN, or any other kind of Neural Network (NN). Originally such a setup of encoder-decoder architecture was first proposed by sutskever et al. in 2014. Both vanilla RNNs and later its variants were found to have multiple limitations like bottleneck problems, sequential processing, teacher forcing, etc. [Additive or Soft attention](https://vijaydaultani.github.io/blog/2021/10/08/demystifying-attention.html) proposed by Bahandau et al. in 2016 solved the bottleneck problem. Similarly, Transformer was introduced by ([Vaswani et al.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)) to mitigate the issue of sequential processing from the original encoder-decoder architectures. 

In a conventional encoder-decoder setup of RNNs, one can utilize the concept of batching to achieve parallelization **across** input and output sequences. But parallelization **within one single** input and output sequence is not possible since recurrent models typically factor computation along the symbol positions of the input and output sequences. This alignment between position and steps in computation time meant RNN could generate a series of hidden states $$h_t$$ as a function of previous hidden state $$h_{t-1}$$ and the input $$x_t$$ for current position $$t$$. This inherent sequential nature to process input symbols one at a time, in an order inhibits parallelization within training examples. For instance, let's consider Fig.1. which represents a conventional encoder-decoder architecture for machine translation task from English to Japanese. In the figure, the encoder generates four hidden states $$h_1, h_2, h_3, h_4$$ corresponding to four input symbols, i.e., "Better", "late", "than", and "never" respectively. Therefore, the hidden state $$h_2$$ which corresponds to the second input symbol "late", is generated based on the previous hidden state $$h_1$$ and the current input symbol $$x_2$$ ("late"). Hence, due to this sequential processing of the symbols by the encoder, it can only generate the next hidden state after generating the hidden state for the previous input symbol.

![seq-to-seq-using-decoder]({{ '/assets/images/demystifying_attention/seq-to-seq-using-attention-example.drawio.png' | relative_url }})
{: style="text-align: center"}
*Fig. 1 Abstractive example of encoder-decoder architecture usage translating an English sentence "Better late than never" to Japanese "決して 遅く ない より 良い".*

The attention mechanism provided a solution for the bottleneck problem of RNN's, i.e., enabling the modeling of dependencies without regard to their distance in the input or output sequence. However, the core problem of sequential processing persisted as attention was used in conjunction with an RNN. Later, to reduce the sequential computation, several works were proposed, e.g., [ByteNet](https://research.google/pubs/pub48560/) and [ConvS2s](http://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf). These two works mainly employed convolution operation as a building block to compute the hidden representations of input and output sequence in parallel, but the number of operations to relate two input and output symbols at arbitrary positions grew with distance; on the contrary, Transformer could keep this constant.

The authors employed the concept of self-attention that can relate different positions of a single sequence to compute a representation of the sequence. In essence, self-attention is very similar to **conventional RNN** and **attention mechanism**. Similar to conventional RNN, self-attention computes a latent representation of the input sequence. However, the conventional RNN will process one symbol at a time; self-attention will process symbols in parallel. And similar to attention mechanism, self-attention will relate, i.e., find similarity between two symbols. However, conventional attention was used only to relate one output with one input symbol; self-attention will additionally relate two input symbols or two output symbols. One should notice that even though the raw ingredients for Transformer, i.e., concepts such as attention, self-attention, etc., existed previously but Transformer was the first model that entirely relied on self-attention to compute the representations of its input and output without using any sequence-aligned RNNs or CNNs. 


## Problem Definition
Formally our goal is to learn an encoder and decoder such that, given an input sequence of symbol representations $$(x_1,...,x_n)$$, the encoder can map the input sequence of symbols to a sequence of continuous representations $$z = (z_1,...,z_n)$$. And later given $$z$$, the decoder can generate output sequence $$(y_1,...,y_n)$$ of symbols. 

## Brief model description

A lot is happening in the Transformers architecture in Fig.2. Let's briefly introduce one concept at a time. The architecture is split into two stacks, i.e., the encoder stack on the left and the decoder on the right. Further, each stack is organized into what is known as layers (also called blocks). Though it is not apparent directly from Fig.2, encoder and decoder stacks consist of **N** (6 in paper) layers stacked on top of each other. These multiple layers are not visible in the diagram for brevity. Next, each encoder and decoder layer is composed of sublayers. The encoder layer comprises two sublayers (**Multi-Head Attention** and **Feed Forward** network). The decoder layer consists of one additional sub-layer, i.e., Masked Multi-Head Attention; therefore, a total of three sublayers (**Masked Multi-Head Attention**, **Multi-Head Attention**, and **Feed Forward** network). You should notice the residual connections around each of the sublayers in the encoder and decoder.

![transfomer-model-architecture]({{ '/assets/images/demystifying_transformer/transformer_model_architecutre.jpg' | relative_url }}){: width="400" }
{: style="text-align: center"}
*Fig. 2 The Transfomer - model architecture*

Next, you should observe that **Input Embedding** is added with **Positional Encoding** and fed as an input to the lowest encoder layer. Similarly, **Output Embedding** added with **Positional Encoding** of output symbols is provided as an input to the lowest decoder layer. Please notice a slight difference on the decoder side where embeddings for **Output** symbols are shifted right by one position before feeding as an input to the decoder layer. Now you should notice the connection from the top of the encoder stack to the multi-head attention sublayer of the decoder. How do those connections work? Well, we will discuss that in detail later in this article. At last, there is a linear layer on the top of the decoder stack, followed by the softmax layer to calculate the probability distribution for the output symbols. Given that we have understood the big picture, let's dive deep into understanding associated concepts in detail one at a time to make sense of the complete architecture later.

{% capture link_section_query_key_value %}{% link _posts/2021-10-10-demystifying-transformer.md %}#query-key-and-value{% endcapture %}
## Query, Key, and Value
Let's switch the gear a bit and discuss an essential concept of **Query (Q)**, **Key (K)**, and **Value (V)** inspired by their usage in Information Retrieval (IR). To give you motivation for the terminologies, let's consider an example of a web search for a topic of interest (e.g., bitcoin). The first step for the web search is to insert our Query (i.e., bitcoin) in a search engine such as Google. Then behind the scenes, the Google search engine will match the input query against the database of all Keys, e.g., documents title. At last, the documents whose keys (i.e., document title) match the input Query are then ranked to show the most relevant Values (e.g., document content) on the top of the search results. The web search is more complicated than presented in the above few lines, but the goal here is to give you the motivation behind the terms Q, K, and V rather than explaining the detailed working of a web search. 

## Attention
Given we know the motivation of Query, Key, and Value, let's recall an independent and essential concept of [attention](https://vijaydaultani.github.io/blog/2021/10/08/demystifying-attention.html). In essence, attention determines how related an input and output symbol are. We calculated this relatedness between the input and output symbols by using their corresponding hidden states as a proxy to estimate the relatedness between them. Once we have calculated this relatedness between one output symbol and all input symbols, we normalize this relatedness using a softmax function to generate normalized attention scores. The attention score tells us the importance of distribution over input symbols to generate an output symbol.

![scaled_dot_product]({{ '/assets/images/demystifying_transformer/scaled_dot_product.png' | relative_url }}){: width="300" }
{: style="text-align: center"}
*Fig.3 Scaled Dot-Product Attention*

In the case of the transformer, we will unite the two concepts of attention and Q, K, V together. An attention function in the case of the transformer can be described as mapping a Query (i.e., output symbol) and a Key-Value (i.e., input symbol) pairs to an output. All query, key, value, and output are vectors. The output of an attention function is computed as a weighted sum of the values, where the weight assigned to each value is computed using a compatibility function of the query with the corresponding key.

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
\label{eq:scaled_dot_attention}
$$

Out of all the different types of attention, the paper's authors decided to use a variation of dot product known as scaled dot product attention as shown in \eqref{eq:scaled_dot_attention}. They call it scaled because once the dot product is calculated, it is divided by $$\sqrt{d_{k}}$$. Where input query $$Q$$ and key $$K$$ are of dimension $$d_{k}$$. We do scaling because the multiplication of query and key can lead to a large magnitude product and therefore push softmax in the regions where it has extremely small gradients. To counteract this effect, they scale the dot product by $$\sqrt{d_{k}}$$. If you have a question, why do we divide by specifically by $$\sqrt{d_{k}}$$ Remember both query and key are of the same dimension, and therefore, it made sense to involve some variation of $$d_{d}$$ but why specifically $$\sqrt{d_{k}}$$, the authors didn't provide any insight. At last, we take the softmax of the product between the query and key to get relative weight distribution across keys. Finally, values are weighted by the output of compatibility calculated between the query and key.  In practice, the attention function is computed in parallel on a batch of queries (Q), keys (K), and values (V) in parallel to make the computation effective using the fast matrix multiplication libraries. 

## Self-Attention
Now let's understand how the concepts of **Attention** and **Q**, **K**, **V** relate to the idea of self-attention in Transformer.

>Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence to compute a representation of the sequence.

As the definition suggests in self-attention we relate all the symbols within a sequence against each other. Again as the word relate suggests what we are interested in calculating is how compatible two symbols are. 

In Section [Query, Key and Value]({{ link_section_query_key_value }}) we 
learned about the inspiration for the concepts of Query, Key, and Value. However, in the case of the transformer architecture, there is a very subtle difference in the usage of Q, K, and V from our previous web search example. In the case of the transformer, either all three, i.e., Q, K, and V, can be all the same (intra-encoder self-attention and intra-decoder self-attention), or K, V can be the same but different from Q (inter encoder-decoder attention). 

## Where is attention used in the model
In transformer architecture, attention is used in total at three places (**encoder only**, **decoder only**, and **across encoder-decoder**). First, usage is within the encoder in the form of encoder self-attention. And in this usage of self-attention, all inputs, i.e., queries, keys, and values, come from the output of the previous encoder layer.

Second, usage is within the decoder in the form of decoder self-attention. Similar to self-attention, in the encoder's case, all of the inputs, i.e., queries, keys, and values, come from the same place, i.e., the output of the previous decoder layer.

Third, usage is across encoder and decoder attention layers. In this case, different from the self-attention in the encoder only and decoder only, the queries come from the previous decoder layer; however, keys and values come from the output of the encoder. This connection between the encoder and decoder allows the queries from the decoder to attend all positions in the input sequence, and it works similarly to attention in conventional sequence-to-sequence RNNs.

### Multi Head Attention
Rather than using a single attention model with queries, keys, and values of $$d_{model}$$ dimension, the authors found it was beneficial to linearly project the queries, keys, and values $$h$$ times with different learned linear projections to $$d_{k}$$, $$d_{k}$$, and $$d_{v}$$ dimensions respectively. Multi-head attention allows the model to jointly attend to information from different representative subspaces at different points. 

![scaled_dot_product]({{ '/assets/images/demystifying_transformer/multi_head_attention.png' | relative_url }}){: width="300" }
{: style="text-align: center"}
*Fig.4 Multi-Head Attention*

$$
\operatorname{MultiHead}(Q, K, V)=\text { Concat }\left(\text { head }_{1}, \ldots, \text { head }_{\mathrm{h}}\right) W^{O}
\label{eq:multi_head_attention}
$$

$$
\text { where head }_{\mathrm{i}}=\text { Attention }\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\label{eq:head_i}
$$

Refer to \eqref{eq:head_i} and \eqref{eq:multi_head_attention} for the formulation of single-head and multi-head attention, respectively. In \eqref{eq:head_i}, the three matrix multiplications $$Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}$$ generate the linear transformation for queries, keys and values respectively followed by scaled dot-product attention function from \eqref{eq:scaled_dot_attention}. It is emperical to observe that Multi-Head Attention in \eqref{eq:multi_head_attention} does not operate batched queries $$Q$$, keys $$K$$, values $$V$$ of $$d_{model}$$ dimension but on their linearly transformed versions with $$d_{k}, d_{k}, d_{v}$$ dimensions respectively.


#### Masked Multi Head Attention
We apply a mask to the output from the product of the query and key both at encoder-only and decoder-only self-attention. In the case of encoder self-attention, this mask is used to mask out the padding values so that they don't participate in attention calculation. Similarly, in the case of decoder self-attention, a mask is applied to hide the future tokens so that they don't interfere with attention scores calculation for self-attention. Also, note similar to conventional encoder-decoder architecture, the decoder was used in autoregressive settings, i.e., the previously generated symbol was consumed as input by the decoder to generate the next symbol.

## Efficiency
In the original content-based attention introduced by Bahandau, the attention score was computed by a simple FFNN based on equation \eqref{eq:attention_score}

$$
e\left(s_{j-1}, h_{i}\right)=v_{a}^{\top} \tanh \left(W_{a} s_{j-1}+U_{a} h_{i}\right)
\label{eq:attention_score}
$$

Where $$h_i$$ is the hidden state from the encoder and $$s_{j-1}$$ is the hidden state from the decoder. The drawback of the above formulation is for m input and n output symbols; one needs to compute the attention scores by running through the FFNN network $$m * n$$ times to find the compatibility between hidden states of the encoder and decoder, which is expensive.

Instead, in the case of a transformer, authors first transform both the input and output symbols into a common space. While avoiding computational heavy function, i.e., FFNN, they calculate the compatibility by using a simple dot product to calculate the compatibility.

## Embedding

### I/P and O/P Embedding
As a common practise in sequence transduction models, the authors used learned embeddings to convert the input symbols and output symbols to vectors of dimension $$d_{model}$$. Also at the decoder side they learned the linear transformation and softmax function to be able to convert the decoders output to predict next-token probabilities.

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
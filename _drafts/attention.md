---
layout: post
comments: true
title:  "Demystifying Attention."
excerpt: "This post will demystify attention mechanism one concept at a time. Not only that, we will try to understand how the attention mechanism interacts with conventional encoder-decoder architecture."
date:   2021-09-23 05:00:00
tags: attention rnn
---
> In this article, I will introduce the most common form of attention (also know as soft-attention or content-based attention), proposed by ([Bahdanau et al.](https://arxiv.org/abs/1409.0473)) to improve the performance of basic encoder-decoder architecture for neural machine translation (NMT). I plan to write all different types of attention in a separate article and therefore the goal of this article is neither to introduce various forms of attention nor differentiate among them. Furthermore, in this article I aim to first provide the motivation, which is followed by the relevant mathematical equations to help deepen the understanding of the topic.

{: class="table-of-content"}
* TOC
{:toc}

## Basic Encoder-Decoder Architecture
A sequence to sequence model was first proposed by ([Sutskever, et al. 2014](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)) in the form of a basic encoder-decoder architecture. What they proposed was a first general end-to-end approach to sequence learning i.e., mapping input sequences to target sequences. To put it simply, an encoder takes in sequence of input and maps that into a context vector which was consumed by a decoder to generate target sequence. In order to appreciate the magical power of attention, it is important to first understand how both encoder and decoder works. Once you understand how this two components work it will be easier to visualize their limitations which will finally lead us to the motivation for attention and how it eliminates those limitations. Ok, then let's jump in to understand how encoder and decoder works!

### Encoder
An encoder in basic encoder-decoder architecture reads the input sequence $$x$$ = $$ x_{1}, x_{2}, .. x_{Tx} $$ and compress that to a vector of fixed-length (dimension) also know as context vector $$c$$. Why? the motivation for this transformation is to map the input sequence to a single vector in the latent space, which is language agnostic so that the decoder can generate target sequence conditioned on this previously generated language agnostic context vector by encoder. Now the questions comes how does encoder does this compression? well the most common approach is to use an RNN such that.

$$
h_{i}=f\left(x_{i}, h_{i-1}\right)
\label{eq:encoder_state}
$$

In \eqref{eq:encoder_state} $$ i $$ represents the time stamp, $$ h_{i} \in \mathbb{R}^{n} $$ is a n-dimensional hidden state of the encoder at i-th timestamp. And $$ f $$ is a nonlinear function for e.g. LSTM. To put it simply \eqref{eq:encoder_state} encoder is computing a encoded representation of all the input seen so far. When encoder has completed reading the input at the end we would like to have a representation which encodes all the input information seen so far and this is how $$ h_{i} $$ will be useful.

Said that, we have now understood how the encoded representation at each time-step of input sequence is calculated by the encoder. Now the question arises, how shall one combine the individual encoded representations to form a context vector i.e. a single piece of information which can be used by decoder to start generating the target sequence? To put it mathematically in \eqref{eq:context_vector} we want a function $$ q $$ which can transform all the encoded representations i.e. $$ h_{1}, h_{2}, .., h_{T_{x}} $$ into a single context vector $$ c $$.

$$
c=q\left(\left\{h_{1}, \cdots, h_{T_{x}}\right\}\right)
\label{eq:context_vector}
$$

Hmm, what should be a good choice of this function $$ q $$. Well in ([Sutskever, et al. 2014](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)) they proposed a logically simple function for $$ q $$. Therefore, they decided to encode variable-length input sequence into a fixed-length vector based on the last encoded representation i.e. $$ h_{T_x} $$.

$$
c=q=h_{Tx}
\label{eq:encoder_context_vector}
$$

Essentially this means the context vector was the same as the last hidden state of the encoder. One might think why did they decided to use only the last hidden state and not the hidden states in the previous steps? Well the motivation behind this formulation is simple, encoder when have already consumed complete input sequence, should have compressed all the essential information from the input sequence down into a single hidden vector after reading the final token in the input sequence. 

Good, so we have learned so far how encoder encodes the input sequence in hidden states and how hidden states are used to determine the context vector $$c$$. Now the next piece in the puzzle is to understand how decoder plans to use this context vector. So let's jump into the details of decoder..

### Decoder
A decoder is initialized with the context vector $$c$$ generated by encoder and generates the target sequence $$y$$. From a probabilistic perspective in \eqref{eq:overall_objective}, the goal is to find a target sequence $$y$$ that maximizes the conditional probability $$y$$ given an input sequence $$x$$ i.e. 

$$
\arg \max_{\mathbf{y}} p(\mathbf{y} \mid \mathbf{x})
\label{eq:overall_objective}
$$

Now the question is how does decoder find such a target sequence $$y$$? Well, The decoder defines the joint probability of generating $$y$$ in the form of ordered conditional probabilities as shown in \eqref{eq:joint_probability_to_ordered_conditional}

$$
p(\mathbf{y})=\prod^{T} p\left(y_{j} \mid\left\{y_{1}, \cdots, y_{j-1}\right\}, c\right)
\label{eq:joint_probability_to_ordered_conditional}
$$

Notice the subtle difference between \eqref{eq:overall_objective} and \eqref{eq:joint_probability_to_ordered_conditional}. Where on one hand in \eqref{eq:overall_objective} we desired to condition on $$x$$, on other hand in \eqref{eq:joint_probability_to_ordered_conditional} we are conditioning on $$c$$ which is acting as a proxy of the complete input sequence. 

Subsequently, the next question is to answer how to calculate the individual conditional probabilities? Let's see how to device this conditional probability of generating target word $$y_j$$? First observation is this conditional probability should consume $$c$$ as the compressed encoded representation of input sequence. Second observation is this conditional probability should take previously generated words. Now, this is the tricky part rather than consuming all previous generated target words, we break this into two parts i.e. $$y_{j-1}$$ target word generated in previous time step and $$s_j$$ the hidden state of the decoder. To put it simply, decoder model each conditional probability for generating $$ y_{j} $$ based on three factors. First, previously generated word $$ y_{j-1} $$, second, current hidden state of the decoder $$ s_{j} $$ and third, the context vector generated by the encoder i.e. $$ c $$. 

$$
p\left(y_{j} \mid\left\{y_{1}, \cdots, y_{j-1}\right\}, c\right)=g\left(y_{j-1}, s_{j}, c\right)
\label{eq:conventional_decoder}
$$

Notice similar to encoder counterpart in \eqref{eq:encoder_state}, in decoder's formulation \eqref{eq:conventional_decoder} $$ g $$ is a non-linear function for e.g. softmax.

So far we understood the motivation and internal working on encoder and decoder with their mathematical formulations. This will bring us on understanding what is indeed the limitation of the above explained encoder-decoder architecture.

## Limitations of Basic Encoder-Decoder Architecture
We saw in \eqref{eq:encoder_context_vector}, in a basic encoder-decoder architecture setting, an encoder maps the complete input sequence to a fixed-length context vector $$c$$. To perform this mapping, the encoder squashes all the information of an input sequence, regardless of its length, into a fixed-length context vector. Furthermore, we also noticed in \eqref{eq:conventional_decoder} a decoder generates target word $$y_j$$ which is conditioned on three factors, two of them i.e. $$y_{j-1}$$ and $$s_j$$ from the decoder side and rest one i.e. $$ c $$ coming from encoder as the condensed version of the input.  Indeed, this squashing into a fixed-length context vector and then conditioning on that single condensed context vector for generating target sequence introduces a serious limitation for the basic encoder-decoder models. 

Why? because a single vector might not be sufficient to represent all the information from an input sequence. This limitation is known as the bottleneck problem, which is exaggerated when dealing with long input sequences, especially input sequence longer than the sentences seen in the training corpus. ([Cho et al.](https://aclanthology.org/W14-4012/)) provided evidence for this problem where they demonstrated the performance of the basic encoder-decoder deteriorates rapidly as the length of the input sequence increases.

## Solution? Decoder with Attention
([Bahdanau et al.](https://arxiv.org/abs/1409.0473)) proposed the solution i.e. attention for the above mentioned limitation of basic encoder-decoder architecture i.e. bottleneck. The motivation behind their solution was rather than relying on a single fixed-length context vector, to use a sequence of context vectors and choose a subset from those adaptively. To put it mathematically, a modified version of the conditional probability introduced in \eqref{eq:conventional_decoder} is defined as 

$$
p\left(y_{j} \mid y_{1}, \ldots, y_{j-1}, \mathbf{x}\right)=g\left(y_{j-1}, s_{j}, c_{j}\right)
\label{eq:attention_decoder}
$$

Let's break down \eqref{eq:attention_decoder} one piece at a time. $$ g $$ is a ? function. This function is dependent on three factors target word generated in previous time step $$ y_{j-1} $$, decoders hidden state from current time step $$s_j$$ and individual context vector from current time step $$c_j$$. The first of three factors $$y_{j-1}$$ is easy to comprehend, now let's focus on rest two $$s_j$$ and $$c_j$$

First, let's focus on $$s_j$$ in \eqref{eq:attention_decoder}. First of all note that we are simply using different variable names to diffentiate the hidden states of encoder ($$ h $$) to that from decoder (i.e. $$ s $$). Also in the  above formulation in \eqref{eq:attention_decoder} $$ s_{j} $$ is decoder hidden state and is further defined as

$$
s_{j}=f\left(s_{j-1}, y_{j-1}, c_{j}\right)
\label{eq:attention_decoder_hidden_state}
$$

Where $$ f $$ is some non-linear function over previous hidden state of decoder $$ s_{j-1} $$, previously generated target word $$ y_{j-1} $$ and individual context vector dedicated to the current time stamp $$j$$  $$ c_{j} $$. 

Coming back to \eqref{eq:attention_decoder} let's try to understand the final factor $$c_j$$. Observe in \eqref{eq:attention_decoder} different from the formulation of basic decoder in \eqref{eq:conventional_decoder} now the conditional probability of each target word $$ y_{j} $$ is based on $$ c_{j} $$ rather than on $$ c$$, therefore each target word will be conditioned on it's individual context vector rather than being conditioned on a single context vector $$ c $$. This individual context vector $$ c_{j} $$ can effectively change for each $$ j $$ and will convey information what part of input sequence is important to generate target word $$ y_{j} $$. 

So far, we have understood what will be the implication of these individual $$ c_{j} $$ which leads us to next question. How to calculate $$ c_{j} $$ in a way that it can help decoder decide which part of input sequence to focus on for generating next target word $$y_j$$. 

The answer is, a simple alignment model. This alignment model should help us determine to generate $$ y_{j} $$ what is the distribution of importance of each input word $$ x_{i} $$ (remember not just the immediate input word but a few neighboring words play a crucial role in generating each hidden representation). Specifically, we want to know how aligned the previous decoder hidden state $$ s_{j-1} $$ is to all of the other input hidden states $$ h_{i} $$. 

![attention_block_diagram]({{ '/assets/images/attention-block-diagram.png' | relative_url }})
{: style="text-align: center"}
*Fig. Block diagram showing input and output to the alignment model. $$h_j$$ represents hidden state of the encoder and $$s_{i-1}$$ represents the hidden state of the decoder*

([Bahdanau et al.](https://arxiv.org/abs/1409.0473)) decided a simple feed-forward neural network for the above alignment model as described in \refeq{eq:attention_score}. The decision to choose such a simple model for alignment was due to it's associated time complexity. Since the alignment model  will be called $$T_{x} \times T_{y} $$ number of times to be able to generate the alignment score between $$h_i$$ and $$s_{j-1}$$. 

$$
e\left(s_{i-1}, h_{j}\right)=v_{a}^{\top} \tanh \left(W_{a} s_{i-1}+U_{a} h_{j}\right)
\label{eq:attention_score}
$$

In the above equation, e (alignment score is also sometimes known as energy) is the function modeled by FFNN, which generates the alignment score. $$ W_{a} $$ and $$ U_{a} $$ are the weight matrices for previous decoder state $$ s_{i-1} $$ and input state in consideration $$ h_{j} $$ respectively. $$ v_{a} $$ is a weight vector that is learned during the training process. Let's dissect the above equation; at heart, it is generating a single vector representation by adding a transformed version of decoders previous state $$ s_{i-1} $$ and input state $$ h_{j} $$ and passing through the $$tanh$$ function, i.e., squashing it between -1 and 1 to represent the fact how much both the states are aligned. -1 representing they are weakly aligned and 1 they are strongly aligned. Notice the generated aligned score, at last, is getting element wise multiplied with $$ v_{a} $$ and therefore, there is no restriction on the value of the final alignment score to be less than 1. 

Remember, so far we have only generated alignment score, which is not a probability and therefore need not be between 0 and 1. Though we can get the relative importance of hidden states by directly comparing the raw alignment scores, we transform this raw score into a probability with the help of our old time friend $$softmax$$ to squash the generated unbounded alignment scores between 0 and 1. Transforming into probability is required because eventually, we want to use this alignment score (b/w 0 and 1) to decide how much to stress on the input word (i.e., $$ e_{ij} \times h_j $$).

$$
\alpha_{i j}=\frac{\exp \left(e_{i j}\right)}{\sum_{k=1}^{T_{x}} \exp \left(e_{i k}\right)} \in 0 < \mathbb{R}  < 1
\label{eq:alignment_weight}
$$

After using softmax, we get $$ \alpha_{i j} $$ which is the alignment weight between hidden state $$ h_{i} $$ for prediction of target word $$ y_{j} $$. Finally, our dynamic context vector should weight each hidden state ( $$ h_{i} $$ ) based on this alignment weight (i.e. $$ \alpha_{i j} $$ ). This is accomplished by simply scaling (i.e. multiplying) n-dimensonal encoded hidden representation of input word $$ h_i $$ with the alignment weight $$ \alpha_{i j} $$ as forumlated below in \eqref{eq:dynamic_context_vector}. 

$$
c_{i}=\sum_{j=1}^{T_{x}} \alpha_{i j} h_{j} \in \mathbb{R}^{n}
\label{eq:dynamic_context_vector}
$$

Well done! we have accomplished an important thing here, when we add together the weighted representation of all the hidden states we get the dynamic context vector, which will now guide decoder which part of the input sequence to focus on.

To help us visualize better how we caulcate the dynamic context vector $$c_{j}$$ from all the input hidden states $$h_1, h_2, .. , h_{T_x}$$ and decoders previous hidden state $$s_{i-1}$$ refer the figure ? below.

![attention_calculation]({{ '/assets/images/attention-calculation.png' | relative_url }})
{: style="text-align: center"}
*Fig. i is an index for decoder and j is an index for encoder.*

So that we understand how to calculate the dynamic context vector $$c_j$$from \eqref{eq:dynamic_context_vector}, now let's try to feed this into \eqref{eq:attention_decoder_hidden_state} and finally into \eqref{eq:attention_decoder} to generate the final probability of target word $$y_j$$. Below fig ?? summarizes this well.

![alignment-model-detailed]({{ '/assets/images/alignment-model-detailed.png' | relative_url }})
{: style="text-align: center"}
*Fig. Details of the alignment model*

Finally, in order to generate the distribution over the target words in the vocabulary we back substitute value of dynamic context vector $$ c_{i} $$ in equation \eqref{eq:attention_decoder_hidden_state} to help us generate decoder hidden state $$ s_i $$ for the current time step. The decoder which in turn is substituted in equation \eqref{eq:attention_decoder} to generate the probability distribution over the target words.

Now, let's try to visualize the modified version of the conditional probability introduced in \eqref{eq:attention_decoder}.

![seq-to-seq-using-decoder]({{ '/assets/images/seq-to-seq-using-attention.png' | relative_url }})
{: style="text-align: center"}
*Fig. Sequence to Sequence Diagram using Attention Module in the Decoder*


### Further details: Let's figure out the reasoning behind matrix shape

$$
W_{a} \in \mathbb{R}^{n \times n}, U_{a} \in \mathbb{R}^{n \times 2 n} \text { and } v_{a} \in \mathbb{R}^{n}
$$

Let's dissect equation \eqref{eq:attention_score} to understand the size of the weight matrix. Input decoder's previous state i.e. $$ s_{j-1} $$ is a vector of size $$ n $$. Similarly input encoded hidden representation is of size $$ 2n $$ since  $$ h_{i}=\left[\vec{i}_{j}^{\top} ; \overleftarrow{h}_{i}^{\top}\right]^{\top} $$.

First, for matrix multiplication feasibility number of columns in $$ W_{a} $$ has to agree with size of input vector $$ s_{i-1} $$, therefore $$ W_{a} $$ dimensions are $$ [ ? \times n ] $$. Similarly, number of columns in $$ U_{a} $$ has to agree with size of input vector $$ h_{j} $$, therefore $$ U_{a} $$ dimensions are $$[? \times 2n] $$. Where we will soon the derive the number of rows in $$ W_{a} $$ and $$ U_{a} $$. 

Notice for vector addition feasibility of $$ W_{a} s_{i-1} + U_{a} h_{j} $$ number of rows in $$ W_{a} $$ and $$ U_{a} $$ should be same. The output vector of this addition is passed via $$ tanh $$ which simply transforms the value in the vector without changing it's size. Finally the output of $$ tanh $$ is multiplied with vector $$ v_{a}^{\top} $$ and therefore should be of same size. For simplicity convinence all these sizes were choosen to be $$ n $$.

## References

[1] ["Attention and Memory in Deep Learning and NLP."](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/) - Jan 3, 2016, by Denny Britz

[2] ["Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"](https://aclanthology.org/D14-1179.pdf)

[3] ["Sequence to Sequence Learning with Neural Networks"](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)

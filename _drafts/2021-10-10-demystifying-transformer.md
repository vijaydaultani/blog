---
layout: post
comments: true
title:  "Demystifying Transformer!"
excerpt: "This post will help demystify transformer mechanism by presenting motivation behind the concepts and explaining the underlying mathematical equations piece by piece. Also, we will learn how does the transformer mechanism interact with conventional encoder-decoder architecture."
date:   2021-10-08 17:00:00
tags: transformer encoder-decoder rnn
---
> In this article, I will introduce the most common form of transformer (also know as soft-transformer or content-based transformer), initially proposed by ([Bahdanau et al.](https://arxiv.org/abs/1409.0473)) to improve the performance of basic encoder-decoder architecture. I plan to write all different types of transformer in a separate article, and therefore the goal of this article will not be to introduce various forms of transformer and differentiate among them. Furthermore, in this article, I aim first to provide the motivation, followed by the relevant mathematical equations to help deepen the understanding of the topic.

{: class="table-of-content"}
* TOC
{:toc}

## Basic Encoder-Decoder Architecture

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

[1] Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio. ["Neural machine translation by jointly learning to align and translate."](https://arxiv.org/pdf/1409.0473.pdf) ICLR 2015.
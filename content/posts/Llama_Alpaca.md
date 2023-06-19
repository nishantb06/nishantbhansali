---
title: "An Overview of LLaMa and Alpaca models"
date: 2023-04-15T13:34:45+05:30
draft: true
---
# The Annotated LLaMA

# Foreword

Welcome to “**The Annotated LLaMA**”.

One of the most brilliant and well-explained articles I have read is [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) and the [Annotated DETR](https://amaarora.github.io/posts/2021-07-26-annotateddetr.html). It introduced **Attention** like no other post. The simple idea was to present an “annotated” version of the paper [Attention is all you need](https://arxiv.org/abs/1706.03762) along with code.

I have tried to do something similar with the LLaMA models without which the commercial use of many Large Language models would not have been possible. 

[arXiv](https://arxiv.org/abs/2302.13971v1) [Official Code](https://github.com/facebookresearch/llama)

# Introduction

Before Llama came there were a series of really good Large Language Models (LLM’s) like Chinchilla, PaLM and GPT-3, only problem they were trained on proprietory data and were not accessible to the public for research or commercial use. 

On February 27, 2023 . Facebook released a set of models called as LLaMA models that were 

1. Perfomance comparable to State-of-the-art LLM models at that time
2. Trained entirely on Publicly available data.
3. Sizes as small as 7B parameter to as Large as 65B parameters
4. Models were available publicly for research purposes only (Not for Commercial Use). Though these models were leaked later.
5. Inference Code was Open sourced , (not the training code 😓)

With these models, they prove that 

1. It is possible to train SOTA LLM’s without the use propprietary and inaccessible datasets, like ??
2. The perfomance of a model as small as 7B parameters will keep on increasing as with the size of the dataset it is trained on. 
    
    > For instance, although [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) recommends training a 10B model on 200B tokens, we find that the performance of a 7B model continues to improve even after 1T tokens.
    > 
3. They wanted to train and optimise a set of models for best possible perfomance at fixed Inference budgets, by training on more tokens than what is typically used

# Datasets Used

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cae1a4ba-4a8c-4d13-96e8-4e50f828a81b/Untitled.png)

These datasets were used for Pretraining of the model, Note that Wikipedia and Books dataset were used in approximately 2 epochs, while other dataset had only 1 epochs. Overall this datasets is of 4.3 TB!!

In short they have trained a large transformer on a large quantity of daya using standard optimizer (AdamW).

### Architecture

Major changes to the Architecture were 

1. Using RMSNorm instead of LayerNorm 
2. Using Rotary Postional Embeddings (which are relative and not absolute)
3. Caching of keys and values during the attention Mechanism
4. SwiGLU activation function

## How to download the Weights on your Machine

Till now the easiest way I have found to download the weights on your machine is using the [pyllama](https://github.com/juncongmoo/pyllama) package. Steps involved in doing so are 

1. Fork, the repository, and `git clone` it to your system. 
2. then install the pyllama package with `pip install pyllama -U`
3. To download the 7B model use `python -m llama.download --model_size 7B`
    
    Here I faced an issue where the download would stop after a few minutes and had to be started again manually. (can be checked using `htop`) . If you face this issue as well then use this shell script . Credits to this [comment](https://github.com/juncongmoo/pyllama/issues/104#issuecomment-1588856820) on this [issue](https://github.com/juncongmoo/pyllama/issues/104). 
    
    ```jsx
    #!/bin/bash
    # Function to handle stopping the script
    function stop_script() {
      echo "Stopping the script."
      exit 0
    }
    
    # Register the signal handler
    trap stop_script SIGINT
    
    while true; do
      # Run the command with a timeout of 200 seconds
      timeout 200  python -m llama.download --model_size $1 --folder model
    
      echo "restart download"
      sleep 1  # Wait for 1 second before starting the next iteration
    # Wait for any key to be pressed within a 1-second timeout
      read -t 1 -n 1 -s key
      if [[ $key ]]; then
        stop_script
      fi
    done
    ```
    
    And using the script with `bash llama_download.sh 7B`
    
4. After successful download the directory will have the following structure 
    
    ![Note that the size of this folder is **13 GB** !!!](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/82e3f3e8-e0c9-4be3-bb29-6c73a5112182/Untitled.png)
    
    Note that the size of this folder is **13 GB** !!!
    
    ![This is what the logs of a Successulf download will look like!](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/036d78e1-a44e-45af-a59b-3e7d22c2ad2f/Untitled.png)
    
    This is what the logs of a Successulf download will look like!
    
5. Another method is to use the bittorrent link given in the readme of the pyllama repository. 

## Other important resources

[chat gpt support](https://chat.openai.com/share/819738b5-1c35-4e1d-9fb8-31c77f09b947) 

https://github.com/facebookresearch/llama/issues/149

https://github.com/Lightning-AI/lit-llama/blob/main/howto/download_weights.md

https://github.com/karpathy/nanoGPT

https://github.com/lightning-AI/lit-llama

https://github.com/Lightning-AI/lit-parrot

# Code deep dive

In the official Code repository of LLama, the first thing that I noticed was that there were only 3 important code files. (This is obviously because they havent included the training code)

1. **llama/generation.py** : This file has a class which creates the pipeline for prompting (running inference) the model. This includes, sampling the top logits, custom stop function, pre and post processing of the input and output. [[code](https://github.com/facebookresearch/llama/blob/main/llama/generation.py)]
2. **llama/tokenizer.py** : Wraps the entencepeice tokenizer in a new class. [[code](https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py)]
3. **llama/model.py** : Holds the code for the transformer models [[code](https://github.com/facebookresearch/llama/blob/main/llama/model.py)]

Lets start with the code for the Tokenizer

## Tokenizer

The job of the tokenizer is to assign an a numeric id to natural text. Which means after “tokenizing” this prompt - "I believe the meaning of life is” it will give us a tensor which looks like 

`[[1, 306, 4658, 278, 6593, 310, 2834, 338]]` . The tokenizer is responsible for both encoding and decoding(coverting numeric id’s) back to natural text. 

As mentioned in the paper, they have used [sentencepeice’s](https://github.com/google/sentencepiece) (Google’s brainchild) implementation of the **Byte-Pair Encoding subword tokenization** algorithm, which means instead of encoding entire words they break it down into smaller syllabus. For example “Tokenizer” may be broken down into “token” and    “-izer” . In this way their vocabulary size is of 32,000 words. Similarly the entire coprus of text data consists of 1.4 Trillion tokens

> We tokenize the data with the byte- pair encoding (BPE) algorithm ([Sennrich et al., 2015](https://arxiv.org/abs/1508.07909)), using the implementation from Sentence- Piece ([Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)). Notably, we split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters
> 

```jsx
from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os

logger = getLogger()
class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size() # 32,000
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
```

The Tokenizer class isn’t all that complicated. The parameter `model_path` holds the path to the `tokenizer.model` file which was downloaded along with the model weights. 

In the given code, BOS (Beginning of Sentence), EOS (End of Sentence), and Pad IDs (Padding IDs) have the following significance:

1. BOS ID: The BOS ID represents the token ID for the "Beginning of Sentence" token. It is used to indicate the start of a sentence or sequence. In the code, the **`encode`** function checks if the **`bos`** flag is True. If it is, the BOS ID is added at the beginning of the tokenized sequence.
2. EOS ID: The EOS ID represents the token ID for the "End of Sentence" token. It is used to indicate the end of a sentence or sequence. In the code, the **`encode`** function checks if the **`eos`** flag is True. If it is, the EOS ID is added at the end of the tokenized sequence.
3. Pad ID: The Pad ID represents the token ID for the "Padding" token. Padding is often used to make all sequences in a batch have the same length. In the code, the Pad ID is retrieved from the SentencePiece model using **`self.sp_model.pad_id()`**. It is typically used during batching and padding sequences to ensure uniform dimensions.

The BOS and EOS IDs help in marking the boundaries of sentences or sequences, which can be useful for various natural language processing tasks such as machine translation, text generation, and language modeling. The Pad ID ensures that sequences are of the same length when batching, which is necessary for efficient computation in deep learning models.

<aside>
💡  Here the values of bos_id, eos_id and pad_id is (1,2 and -1 correspondingly). Can be checked after creating an object of the Tokenizer class.

</aside>

### Byte pair encoding algorithm (Not Necessary to understand LLaMA)

- Personally, this a really smart,simple algorithm!!
- First the entire corpus is divided into indiviudal characters and a counter is attached to each word, which indicates how many times the word has appeared in the corpus. Each character is now already a part of the Final Vocabulary
- Then each word is divided into its characters and the pairwise occurence of each consecutive characters is counted. The most frequently occuring pair is added to the final corpus. In the sea of characters, wherever these 2 characters were occuring is combined into one and the process of counting the occurences of each pair is repeated and the most frequent one is added to the final Vocab.
- Honestly, just watch [this](https://www.youtube.com/watch?v=HEikzVL-lZU) video by hugging face if you didnt understand my explaination 😟

## Generator

## Model

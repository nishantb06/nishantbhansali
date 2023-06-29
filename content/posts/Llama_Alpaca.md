---
title: "The Annotated LLaMA"
date: 2023-04-15T13:34:45+05:30
draft: true
---

# Foreword

Welcome to “**The Annotated LLaMA**”.

One of the most brilliant and well-explained articles I have read is [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) and the [Annotated DETR](https://amaarora.github.io/posts/2021-07-26-annotateddetr.html). It introduced **Attention** like no other post. The simple idea was to present an “annotated” version of the paper [Attention is all you need](https://arxiv.org/abs/1706.03762) along with code.

I have tried to do something similar with the LLaMA models by Meta Research, without which the commercial use of many Large Language models would not have been possible. 

[arXiv](https://arxiv.org/abs/2302.13971v1) , [Official Code](https://github.com/facebookresearch/llama)
![llama](https://images.unsplash.com/photo-1589182337358-2cb63099350c?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&w=4800)

# Introduction

Before Llama came there were a series of really good Large Language Models (LLM’s) like Chinchilla, PaLM and GPT-3, only problem they were trained on proprietory data and were not accessible to the public for research or commercial use. 

On February 27, 2023 . Facebook released a set of models called as LLaMA models that had 

1. Perfomance comparable to State-of-the-art LLM models at that time
2. Trained entirely on Publicly available data.
3. Sizes as small as 7B parameter to as Large as 65B parameters
4. **Models were available publicly for research purposes** only (Not for Commercial Use). Though these models were leaked later.
5. Inference Code was Open sourced , (not the training code 😓)

With these models, they prove that 

1. It is possible to **train SOTA LLM’s without the use propprietary and inaccessible datasets**.
2. The perfomance of a model as small as 7B parameters will keep on increasing as with the size of the dataset it is trained on. 
    
    > For instance, although [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) recommends training a 10B model on 200B tokens, we find that the **performance of a 7B model continues to improve even after 1T tokens.**
    > 
3. They wanted to train and optimise a set of models for **best possible perfomance at fixed Inference budgets,** by training on more tokens than what is typically used

## LLM Scaling Law’s

Before we dive into training, it’s important to cover how LLMs scale. Understanding scaling lets us effectively balance the size and complexity of your model and the size of the data you’ll use
to train it.

Some relevant history here: OpenAI originally introduced “the LLM scaling laws” in 2020. They **suggested that increasing model size was more important than scaling data size**. This held for about two years before DeepMind suggested almost the polar opposite: that **previous models were significantly undertrained** and that increasing your foundational training datasets actually
leads to better performance.

That changed in 2022. Specifically, DeepMind put forward an alternative approach in their Training Compute-Optimal Large Language Models paper. They found that **current LLMs
are actually significantly undertrained**. Put simply: these large models weren’t trained on nearly enough data.

Deepmind showcased this with a model called Chinchilla, which is a fourth the size of the Gopher model above but trained on 4.6x more data. At that reduced size but with far more training
data, Chinchilla outperformed Gopher and other LLMs.

DeepMind claims that **the model size and the number of training tokens should instead increase at roughly the same rate to achieve optimal performance.** If you get a 10x increase
in compute, you should make your model 3.1x times bigger and the data you train over 3.1x bigger; if you get a 100x increase in compute, you should make your model 10x bigger and your data 10x bigger.

## Datasets Used

![Datasets used](https://i.imgur.com/DhWpc17.png)

These datasets were used for Pretraining of the model, Note that Wikipedia and Books dataset were used in approximately 2 epochs, while other dataset had only 1 epochs. Overall this datasets is of 4.3 TB!!

In short they have trained a large transformer on a large quantity of daya using standard optimizer (AdamW).

## Architecture

Major changes to the Transformer Architecture were 

1. Using RMSNorm instead of LayerNorm 
2. Using Rotary Postional Embeddings (which are relative and not absolute)
3. Caching of keys and values during the attention Mechanism
4. SwiGLU activation function

# How to download the Weights on your Machine

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
    
    ![Note that the size of this folder is **13 GB** !!!](https://i.imgur.com/LsIuE3c.png)
    
    Note that the size of this folder is **13 GB** !!!
    
    ![This is what the logs of a Successulf download will look like!](https://i.imgur.com/YnhC8gG.png)
    
    This is what the logs of a Successulf download will look like!
    
5. Another method is to use the bittorrent link given in the readme of the pyllama repository. 

Size of the 7B model is 13GB (Unquantized). and running `nvidia-smi` told me that GPU memory usage is 13266MiB. Which is roughly 13.3 GB

- chatgpt response to “**My GPU memory usage is 13266MiB. How to interpret this”**
    
    The GPU memory usage of 13266MiB indicates that a portion of the memory on your graphics processing unit (GPU) is currently in use. Here's how you can interpret this information:
    
    1. GPU Memory: GPUs have their own dedicated memory (VRAM) that is used to store data for processing graphics-related tasks. This memory is separate from the system memory (RAM) used by your computer's CPU. The GPU memory usage refers to the amount of VRAM being utilized at a given time.
    2. Unit of Measurement: "MiB" stands for mebibyte, which is a binary multiple of a megabyte. It is often used in reference to memory sizes. One mebibyte is equal to 1,048,576 bytes.
    3. Memory Usage: In your case, the GPU memory usage is 13266MiB, which means that approximately 13,266 mebibytes or roughly 13.3 gigabytes (GB) of the GPU's memory is currently in use.
    4. Implications: The GPU memory usage indicates the amount of memory that your GPU is actively utilizing to store data for ongoing graphics-related operations. It could include textures, shaders, buffers, or other resources needed for rendering images, running computations, or performing other GPU-accelerated tasks.
    5. Considerations: The amount of GPU memory usage can vary depending on the applications or processes running on your system. Graphics-intensive tasks such as gaming, video editing, 3D rendering, or machine learning can consume a significant amount of GPU memory. Monitoring GPU memory usage can help you understand how much memory your applications or workloads require and whether you are approaching the limits of your GPU's available memory.
    6. Limits and Optimization: GPUs have a maximum memory capacity, and if the memory usage exceeds this limit, it can lead to performance issues or even crashes. It's important to ensure that your GPU has enough available memory for the tasks you want to perform. If you encounter limitations due to memory constraints, you may need to optimize your application, reduce the resolution of textures, or consider upgrading to a GPU with higher memory capacity.
    
    Overall, the GPU memory usage of 13266MiB indicates that a significant portion of your GPU's memory is currently being utilized by active processes or applications.
    

# Code deep dive

In the official Code repository of LLama, the first thing that I noticed was that there were only 3 important code files. (This is obviously because they havent included the training code)

1. **llama/generation.py** : This file has a class which creates the pipeline for prompting (running inference) the model. This includes, sampling the top logits, custom stop function, pre and post processing of the input and output. [[code](https://github.com/facebookresearch/llama/blob/main/llama/generation.py)]
2. **llama/tokenizer.py** : Wraps the entencepeice tokenizer in a new class. [[code](https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py)]
3. **llama/model.py** : Holds the code for the transformer model [[code](https://github.com/facebookresearch/llama/blob/main/llama/model.py)]

Lets start with the code for the Tokenizer

## Tokenizer

The job of the tokenizer is to assign an a numeric id to natural text. Which means after “tokenizing” this prompt - "I believe the meaning of life is” it will give us a tensor which looks like 

`[[1, 306, 4658, 278, 6593, 310, 2834, 338]]` . The tokenizer is responsible for both encoding and decoding(coverting numeric id’s) back to natural text. Also keep in mind that there is a tradoff between vocab size and the sequence length for a an input. What we want is a small enough vocab size which is representative of the entire corpus but also keeps the sequence length small. A large sequence lenght would mean more memory of the GPU consumed by the input!

As mentioned in the paper, they have used [sentencepeice’s](https://github.com/google/sentencepiece) (Google’s brainchild) implementation of the **Byte-Pair Encoding subword tokenization** algorithm, which means instead of encoding entire words they break it down into smaller syllabus. For example “Tokenizer” may be broken down into “token” and    “-izer” . In this way their vocabulary size is of 32,000 words. Similarly the entire coprus of text data consists of 1.4 Trillion tokens

> We tokenize the data with the byte- pair encoding (BPE) algorithm ([Sennrich et al., 2015](https://arxiv.org/abs/1508.07909)), using the implementation from Sentence- Piece ([Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)). Notably, we split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters
> 

```python
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

## Model

Before we dive deeper into the architecure and the working of the LLaMA transformer model,It is important to know what the model args for the 7B parameter look like:

```python
ModelArgs(
	dim=4096, # An internal dimension for the transformer architecture
	n_layers=32, # Number of transformer blocks
	n_heads=32, # Number of heads
	vocab_size=32000, # Vocab size of the tokenizer
	multiple_of=256, # A param used in calculating the positional embeddings
	norm_eps=1e-06, # eps value used when dividing by zero during Normalisation
	max_batch_size=1, # Max batch size during inferencing
	max_seq_len=1024, # input to the transformer model can be this long only
)
```

 In the code these params are stored in the `params.json` file, that was downloaded along with the checkpoint path

```python
import json
max_seq_len = 1024
max_batch_size = 1

with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

# create a model args object
# ModelArgs is a a simple dataclass that contains the parameters for the model
model_args: ModelArgs = ModelArgs(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
)
model_args.vocab_size = tokenizer.n_words

# model is loaded through the checkpoint by 
# ckpt_dir is the path to the `7B` directory in the folder downloaded from the llama.download filea
checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
ckpt_path = checkpoints[0]
checkpoint = torch.load(ckpt_path, map_location="cpu")

model = Transformer(model_args)
model.load_state_dict(checkpoint, strict=False)
```

After the model is loaded you can count the parameters by 

```python
# count number of parameters in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count_parameters(model)
# Output
# 6_738_415_616 (6.7 Billion)
```

Now, We can see that in the [llama/model.py](https://github.com/facebookresearch/llama/blob/main/llama/model.py) file, there are 5 Classes → 

1. [RMSNorm()](https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#LL33C7-L33C14) : Defines the Normalisation technique used in the model
2. [Attention()](https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L76) : Has the code for the MultiHead Self Attention Module
3. [FeedForward ()](https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L153) : Has the code for the MLP attached after each attention layer
4. [TransformerBlock()](https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L178) : Defines one transformer Block, there are total 32 of these in the architecture
5. [Transformer()](https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L198) : Has the embedding layer and code for instantiating the TransformerBlocks

We’ll go over each of these in detail

### Transformer()

There are several variants of transformer language models. LLaMA is an **autoregressive, decoder-only transformer** language models, such as GPT-3. Auto regressive because this model predicts future values of a variable based on its own past values (A term taken from time series analysis). Decoder and Encoder are two parts of the transformer architecture. Encoder is used for building a good understanding of the data, whereas decoder is used for more supervised task like, next token prediction, which is what the LLaMA model does. It is one big model that predicts the next token based on the prompt (context) that we give it.

This encoder starts with an Embedding Layer followed by Multiple residual blocks of Multi Head Self Attention and Feed Forward layers. Each residual block consists of an attention layer, followed by an MLP layer. Both the attention and MLP layers each “read” their input from the residual stream (by performing a linear projection), and then “write” their result to the residual stream by adding a linear projection back in. Each attention layer consists of multiple heads, which operate in parallel. RMS Normalisation technique is used before any attention layer or feed forward layer. 

- Complete Code for Transformer()
    
    ```python
    class Transformer(nn.Module):
        def __init__(self, params: ModelArgs):
            super().__init__()
            self.params = params # ModelArgs
            self.vocab_size = params.vocab_size # 32_000
            self.n_layers = params.n_layers # 32
    
            self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim) # (32_000, 4096)
    
            self.layers = torch.nn.ModuleList()
            for layer_id in range(params.n_layers):
                self.layers.append(TransformerBlock(layer_id, params))
    
            self.norm = RMSNorm(params.dim, eps=params.norm_eps) # shape of output is same as input
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False) # (4096, 32_000)
    
            self.freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
                # 4096 // 32 = 128, 1024 * 2
            ) # torch.Size([2048, 64])
    
        @torch.inference_mode()
        def forward(self, tokens: torch.Tensor, start_pos: int):
            _bsz, seqlen = tokens.shape # (1,8)
            h = self.tok_embeddings(tokens) # (1,8,4096)
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen] # torch.Size([1024, 64])
    
            mask = None
            if seqlen > 1:
                mask = torch.full(
                    (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
                ) # (1,1,8,8) , filled with -inf
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
                # (1,1,8,8) , filled with -inf, but only the upper triangle, lower triangle is 0
                # diagnol = start_pos + 1, so the first 8 tokens are not masked, it basically pushes the diagonola above
    
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
            h = self.norm(h) # (1,8,4096)
            output = self.output(h[:, -1, :])  # only compute last logits # (1, 4096) * (4096, 32_000) = (1, 32_000)
            return output.float() # (1, 32_000)
    ```
    

The input to the transformer forward() function will always be a 2D tensor (batch-size,seq_len), in our case (1,8) . The batch size will always be fixed (1 for inferencing). For simplicity, lets take the batch size as 1 from here on. Seq-length will always keep on varying depending on the prompt, but after passing the prompt throught the transformer once, we only need to give it the next token predicted until the EOS token is predicted. So after the first iteration sequence length will always be 1. For now lets take the `seq_length` to be 1

The Emebdding layer ( self.tok_embeddings )is simply responsible for selecting the embeddings based on the indices passed as input to it. Therefore its size will always be (batch_size,seq_length,params.dim).  In our case, the output of the embedding layer becomes `(1,8,4096)` Note that the internal dimension used for 7B model is 4096. 

Lets look at the positional Embeddings used in this LLaMA architecture now. 

**Rotary Positional Embeddings**

`self.freq_cis` is a tensor (fixed and not a trainable parameter) given by the `precompute_freq_cis` function. Instead of using absolute or trainable positional encodings, LLaMA uses Rotary positional encodings, which are know to have faster convergence and better results on various sequence modelling tasks. 

Rotary encoding transforms pairs of features by rotating in the 2D plane. That is, it organizes the *d* features as 2*d* pairs. Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it by an angle depending on the position of the token.

For now its enough to understand that the function returns a 2D tensor of shape (2048,64) or (2*max_seq_length , params.dim // n_heads), where each element is of the form
![iota3](https://i.imgur.com/tPHF68t.png)
Here `theta` is given by 

![iota2](https://i.imgur.com/6RifwFd.png)

- Code for generating positional embeddings
    
    ```python
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis
    ```
    
- Here is a chatGPT generated response to what the `precompute_freq_cis()` does step by step
    
    The function **`precompute_freqs_cis`** is used to precompute frequency sinusoids (**`freqs_cis`**) that are later used in the transformer architecture. These sinusoids play a role in applying rotational embeddings to the query and key representations, which is a technique used to introduce positional information to the transformer model. Let's break down the function step by step:
    
    1. **`freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))`**:
        - **`torch.arange(0, dim, 2)[: (dim // 2)]`** generates a tensor of values **`[0, 2, 4, ..., dim-2]`**. It selects only the first **`dim // 2`** elements.
        - **`torch.arange(0, dim, 2)[: (dim // 2)].float() / dim`** divides the selected tensor by **`dim`** to obtain a range of values between 0 and 1.
        - **`(theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))`** raises **`theta`** to the power of each element in the range of values, resulting in a tensor of frequency factors.
        - **`1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))`** computes the reciprocal of each element in the tensor, resulting in a tensor of frequencies.
    2. **`t = torch.arange(end, device=freqs.device)`**:
        - **`torch.arange(end)`** generates a tensor with values ranging from 0 to **`end-1`**.
        - **`device=freqs.device`** ensures that the generated tensor is placed on the same device as the **`freqs`** tensor.
    3. **`freqs = torch.outer(t, freqs).float()`**:
        - **`torch.outer(t, freqs)`** computes the outer product between **`t`** and **`freqs`**, resulting in a 2D tensor where each element is the product of corresponding elements from **`t`** and **`freqs`**.
        - **`freqs = torch.outer(t, freqs).float()`** converts the resulting tensor to the **`float`** data type.
    4. **`freqs_cis = torch.polar(torch.ones_like(freqs), freqs)`**:
        - **`torch.ones_like(freqs)`** creates a tensor with the same shape as **`freqs`** and fills it with ones.
        - **`torch.polar(torch.ones_like(freqs), freqs)`** converts the ones tensor and **`freqs`** tensor to polar coordinates, resulting in a complex tensor **`freqs_cis`** where the magnitude is 1 and the phase is determined by the corresponding values in **`freqs`**.
    5. **`return freqs_cis`**: The function returns the computed frequency sinusoids **`freqs_cis`** as the final output.
    
    Overall, the **`precompute_freqs_cis`** function is used to generate frequency sinusoids that are used in the transformer architecture to introduce positional information through rotational embeddings. These embeddings play a crucial role in capturing the relative positions of tokens within the input sequences, allowing the transformer to handle sequential data effectively.
    

For better understanding refer to this annotated blog on [Rotaray Positional Embeddings](https://nn.labml.ai/transformers/rope/index.html) and the [original paper](https://arxiv.org/abs/2104.09864).  

As to how these positional embeddings are used in the attention architecture, we’ll come back to it later.

The embeddings for the sequence, along with it’s mask and starting position are fed to the 32 consecutive `TransformerBlock()` layers.  One thing to note here is that the input and output to the Attention layer  and the feed forward layer are always the same shape. Attention mechanism can be though of as how much more expressive you can make your sequence after each operation. Also Attention and feed forward layer are residual layers, which means that the the input to the attention layer is always added back to the output before they are collectively sent to the next attention layer. 

This is call Residual or Skip Connections and are done so as to allow the gradients to flow back more easily.  [[youtube video of Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s) explaining this concept]

After passing the input through all the `TransformerBlocks()` layers, one final normalisation (RMSNorm) is done before using a Linear layer to predicting the next token for the given input sequence. The Linear layer is of shape `(4096,32_000)`.

Only the last logits  `h[:, -1, :]` to the input sequence is used as. input the the linear layer. With the laws of matrix multiplication we can see that the output of the entire Transformer architecture will always be a tensor of shape `(1,32_000)` . `(1, 4096) * (4096, 32_000) = (1, 32_000)`. These logits represent the model's confidence scores for each token being the next token in the sequence.

The logits produced by the output layer are typically passed through a softmax function to convert them into probabilities. The softmax function normalizes the logits and assigns a probability distribution over the vocabulary. The next token prediction is made by selecting the token with the highest probability as the predicted next token.

The decision to use only the last logits as input to the output layer is a common practice in many sequence-to-sequence tasks. The intuition behind this is that the last hidden state of the sequence, after undergoing multiple layers of attention and transformation, is expected to contain the most relevant and comprehensive information for generating the final output.

By selecting the last hidden state, the model effectively focuses on the most recent context and conditions the output generation on the entire input sequence while taking advantage of the context modeling capabilities of the Transformer decoder.

However, it's important to note that without the full context of the previous hidden states, the model may lose some information that could potentially be useful for certain tasks. Different model architectures or downstream tasks may require different strategies for leveraging the entire sequence of hidden states, and the choice of using only the last logits depends on the specific requirements of the task at hand.

Now lets dive deeper into the class `TransformerBlock()`

### TransformerBlock()

- Complete Code
    
    ```python
    class TransformerBlock(nn.Module):
        def __init__(self, layer_id: int, args: ModelArgs):
            super().__init__()
            self.n_heads = args.n_heads
            self.dim = args.dim
            self.head_dim = args.dim // args.n_heads
            self.attention = Attention(args)
            self.feed_forward = FeedForward(
                dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
                # 4096, 4 * 4096, 256
            )
            self.layer_id = layer_id
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
        def forward(
            self,
            x: torch.Tensor, # (1,8,4096)
            start_pos: int, # 0 (initially)
            freqs_cis: torch.Tensor, # (8, 64)
            mask: Optional[torch.Tensor], # (1,1,8,8)
        ):
            # this is a skip connection
            h = x + self.attention.forward(
                self.attention_norm(x), start_pos, freqs_cis, mask
                # (1,8,4096), 0, (1024, 64), (1,1,8,8)
            ) # (1,8,4096)
            out = h + self.feed_forward.forward(self.ffn_norm(h))
            return out # (1,8,4096)
    ```
    

This class just encapsulates the attention layer and feed forward layer and the code for residual connection can also be seen here `(h = x + attention(x))`  and `out = h + feed_forward(h)`

A few things to note here

- Normalisation (**RMSNorm**) is done to the input before it enters either the attention or the feed forward layer. In the original transformer paper, it was done after the layer.

### RMSNorm()

- **Complete Code**
    
    ```python
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
    
        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
        def forward(self, x):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
    ```
    
- This Class performs the RMS normalization on the input tensor **`x`**.
- It computes the square of **`x`** using **`x.pow(2)`**, calculates the mean along the last dimension using **`mean(-1, keepdim=True)`**, and adds **`self.eps`** to the mean value for numerical stability.
- Then, it applies reciprocal square root (**`rsqrt`**) to the sum of the mean and **`self.eps`**.
- Finally, it multiplies **`x`** with the reciprocal square root to normalize the input.
- In short it is dividing the input by its L2 Norm of the last dimension (4096)
- The normalization is computed using a learnable parameter **`self.weight`** and the **`eps`** value to ensure numerical stability. The module helps to normalize the input data and can be useful in certain neural network architectures to improve model performance and convergence.

### FeedForward()

- Complete Code
    
    ```python
    class FeedForward(nn.Module):
        def __init__(
            self,
            dim: int, # 4096
            hidden_dim: int, # 4 * 4096 = 16384
            multiple_of: int, # 256
        ):
            super().__init__()
            hidden_dim = int(2 * hidden_dim / 3) # 2 * 16384 / 3 = 10922
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) # 256 * (10922 + 256 - 1) // 256 = 11177
    
            self.w1 = nn.Linear(dim, hidden_dim, bias=False) # (4096, 11177)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False) # (11177, 4096)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False) # (4096, 11177)
    
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x)) # (1,8,4096)
    ```
    

In summary, the **`FeedForward`** module applies a series of linear transformations to the input tensor **`x`** using three linear layers (**`self.w1`**, **`self.w2`**, and **`self.w3`**) with appropriate dimensions. The intermediate activation function **`F.silu`** ([Sigmoid Linear unit](https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html)) is applied element-wise to introduce non-linearity. 

### Attention()

- Complete Code
    
    ```python
    class Attention(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
    
            self.n_local_heads = args.n_heads // 1 # 32 // 1 = 32
            self.head_dim = args.dim // args.n_heads # 4096 // 32 = 128
    
            self.wq = nn.Linear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
            ) # (4096, 4096)
            self.wk = nn.Linear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
            ) # (4096, 4096)
            self.wv = nn.Linear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
            )  # (4096, 4096)
            self.wo = nn.Linear(
                args.n_heads * self.head_dim,
                args.dim,
                bias=False,
            ) # (4096, 4096)
            self.cache_k = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
                # (1,1024,32,128)
            )
            self.cache_v = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
                # (1,1024,32,128)
            )
            if hiq.get_env_bool("KV_CAHCHE_IN_GPU", True):
                self.cache_k = self.cache_k.cuda()
                self.cache_v = self.cache_v.cuda()
    
        def forward(
            self,
            x: torch.Tensor, # (1,8,4096)
            start_pos: int, # 0 (initially)
            freqs_cis: torch.Tensor,  # (8, 64)
            mask: Optional[torch.Tensor],  # (1,1,8,8)
        ):
            bsz, seqlen, _ = x.shape
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            # all of shape (1,8,4096)
    
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim) # (1,8,32,128) # (1,1,32,128)
            xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim) # (1,8,32,128) # (1,1,32,128)
            xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim) # (1,8,32,128) # (1,1,32,128)
    
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis) # (1,8,32,128), (1,8,32,128)
            # # (1,1,32,128), (1,1,32,128)
    
            self.cache_k = self.cache_k.to(xq) # (1,1024,32,128) moved to the same device as xq
            self.cache_v = self.cache_v.to(xq) # (1,1024,32,128) moved to the smae device as xq
    
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk # (1,1024,32,128)
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv # (1,1024,32,128)
    
            keys = self.cache_k[:bsz, : start_pos + seqlen] # (1,start_pos + seqlen,32,128) or (1,8,32,128)
            values = self.cache_v[:bsz, : start_pos + seqlen] # (1,start_pos + seqlen,32,128) or (1,8,32,128)
            # in the next run it will be (1,9,32,128) 
    
            xq = xq.transpose(1, 2) # (1,32,8,128) -> (1,32,1,128)
            keys = keys.transpose(1, 2) # (1,32,8,128) -> (1,32,9,128)
            values = values.transpose(1, 2) # (1,32,8,128) -> (1,32,9,128)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # (1,32,8,8)
            # matrix multiply of (1,32,8,128) and (1,32,128,8) resulting in # (1,32,8,8)
            # matrix multiply of (1,32,1,128) and (1,32,128,9) resulting in # (1,32,1,9)
            if mask is not None: # (1,1,8,8) -> (1,1,1,1)
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen) # (1,32,8,8) -> (1,32,1,9)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq) # (1,32,8,8) -> (1,32,1,9)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim) # (1,32,8,128)
    		# matrix multiply of (1,32,8,8) and (1,32,8,128) resulting in (1,32,8,128)
            # matrix multiply of (1,32,1,9) and (1,32,9,128) resulting in (1,32,1,128)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # (1,8,4096) -> (1,1,4096)
    
            return self.wo(output) # (1,8,4096) x (4096,4096) -> (1,8,4096)
            # (1,1,4096) x (4096,4096) -> (1,1,4096)
    ```
    

**The Multi Head Self Attention mechanism**

The whole point of the Self Attention operator is to generate a better representation of input sequence, that means the token/embedding a the nth position in the sequence should have condensed the knowledge of the sequence coming before it. One example for this is that the token at the nth position can be the average  or some sort weighted average of all the tokens coming before it. 

In the self attention mechanism, we want to gather information from the past but we want to do it in a data dependent way we first generate a query,key,value from each token. This is done by multiplying each token by 3 separate weight matrices. In this way one token is converted into 3 representations with the help of a linear layer. 

Now we pick one query vector and take the scaled dot product of it with each key vector. We then multiply this scalar with its value vector. All of these value vectors are then added to get the representation of the token for which the query vector was for. 

Can you see how amazing this is!!. I interpret this as the new representation has all of the value vectors weighted as per the relevant information  it contains. It literally chooses what to include  and how much knowledge  to include in itself. Query vector can be though of as what I’m looking for, key vector is what do I contain, value vector is if you find me intersting, here’s what I will communicate to you. 

Multihead attention is just creating `n_head` different q,k,v vectors and letting them operate in parrallel. Just like using more than one filters in CNN’s, so that different info and context are learnt in each head. Later all the output is concatenated and multiplied with another weight matrix, which is fed as input to the next attention layer. 

<aside>
💡 It’s not alsways necessary to not let the current token look at the future tokens, like in encoders or sentiement analysis. That is why here we use masking so that the value vectors for the tokens ahead in the sequence are not added to the its value representation.

</aside>

Other excellent resouces to understand Attention mechanism are 

- [Karpathy’s NanoGPT tutorial](https://youtu.be/kCc8FmEb1nY)
- [The annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [In depth transformers](https://transformer-circuits.pub/)
- [Jay Allamar’s Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

**Caching in the attention mechanism.** 

This was a concept introduce in [this](https://arxiv.org/abs/2112.05682) paper which claims that the attention mechanism does not have to be O(n^2) Operation, instead if we store the key and value vectors in a cache generated in this sequence, we wont have to compute them again when we pass in the next input of the token. 

In the paper they have said

> .First, we use an efficient implementation of the causal multi-head attention to reduce memory usage and runtime. This imple- mentation, available in the [xformers](https://github.com/facebookresearch/xformers) library,2 is inspired by [Rabe and Staats (2021)](https://arxiv.org/abs/2112.05682) and uses the backward from Dao et al. (2022). This is achieved by not storing the attention weights and not com- puting the key/query scores that are masked due to the causal nature of the language modeling task.
> 

As you can see here also with these lines we simply query the cache to get the keys and values from the past. 

```python
keys = self.cache_k[:bsz, : start_pos + seqlen] # (1,start_pos + seqlen,32,128) or (1,8,32,128)
values = self.cache_v[:bsz, : start_pos + seqlen] # (1,start_pos + seqlen,32,128) or (1,8,32,128)
# in the next run it will be (1,9,32,128)
```

**Masking in attention mechanism.** 

During each forward run, if the seq_length passed in is more than 1, a mask is generated which is a tensor of shape (1,1,seq_length,seq_length). In this tensor the upper triangle is filled with `-inf` and the lower traingle is filled with zero’s. Then when the weight matrix is calculated (by multiplying keys and query’s) this mask is added to the weight matrix. In this way the upper traingle of the weight matrix also becomes zero. Then when we take the softmax of this weight matrix, the `-inf` will convert to zeros and the rest of the values will be converted to probability values. 

This is done because we do not want the token at the nth position to contain information from the tokens ahead of itself.  For predicting the next token,the embeddings after the starting_pos should not be “visible” to the previous_tokens . This mathematical trick is explained clearly here in [Karpathy’s tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2533s) on GPT. 

**Use of Rotary positional embeddings**

```python
def apply_rotary_emb(
    xq: torch.Tensor, # (1,8,32,128)
    xk: torch.Tensor, # (1,8,32,128)
    freqs_cis: torch.Tensor, # (8,64)
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # (1,8,32,128) -> (1,8,32,64,2) -> (1,8,32,64)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) # (1,8,32,128) -> (1,8,32,64,2) -> (1,8,32,64)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # (1,8,1,64)
    # Element-wise multiplication of 2 matrices which has complex numbers
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # (1,8,32,64,2) -> (1,8,32,128) 
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) # (1,8,32,64,2) -> (1,8,32,128)
    return xq_out.type_as(xq), xk_out.type_as(xk) # (1,8,32,128), (1,8,32,128)
```

As we know that the positional embeddings are tensors of complex number of the form cos(theta) + i sin(theta). Here we first convert the query and key vectors into their complex forms by reshaping them, and then element wise multiplication with the postional embeddings is done. Then these complex numbered tensor is again converted to their real form and returned. 

## Generation

At the crux of the [generate.py](http://generate.py) is this logic

```python
bsz = len(prompts) # 1
params = self.model.params # those same ModelArgs
assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
# [[1, 306, 4658, 278, 6593, 310, 2834, 338]]

min_prompt_size = min([len(t) for t in prompt_tokens]) # 8
max_prompt_size = max([len(t) for t in prompt_tokens]) # 8

total_len = min(params.max_seq_len, max_gen_len + max_prompt_size) # 264

tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
# a tensor of size (1, 264) filled with -1's

for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = torch.tensor(t).long()
    # fill the first 8(length of prompt_tokens[0]) tokens of tokens with the prompt tokens
input_text_mask = tokens != self.tokenizer.pad_id # a tensor of size (1, 264) filled with True's ,
# where tokens is not -1, other wise False
start_pos = min_prompt_size # 8
prev_pos = 0
for cur_pos in range(start_pos, total_len):
    i = tokens[:, prev_pos:cur_pos]
    logits = self.model(i, prev_pos)
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1)
    next_token = next_token.reshape(-1)
    # only replace token if prompt has already been generated
    next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    )
    tokens[:, cur_pos] = next_token
    prev_pos = cur_pos
    
    if self._should_stop(tokens, prompt_tokens, stop_ids, stop_words):
        break
```

`prompts` as a list of strings will be given to the `generate` function. For now let the prompt be “**I believe the meaning of life is**” . `prompt_tokens are generated through the tokenizer's encode function and the minimum and the maximum prompt size in the entire list of strngs are calculated. For now both of those will be `8` 

Maximum sequence length that can be passed to the transformer is `1024` (A fixed predecided hyper parameter. This is also called the context length. Recent advances have led to the maximum context length becoming 100k as well). Also we do not want the output to be more than 256 tokens. (Again a predetermined fixed hyper parameter) 

Considering these 2 factors in mind the total length that will be generated is calculated by taking minumum of `1024` and `256 + prompt_length`  . Here this will be `264`

A tensor where the intial few values are our original `prompt_tokens` and the rest are filled with `pad_id (-1)` is created. Correspondingly a `input_mask` of size (1, 264) filled with True's , where tokens is not -1, other wise False. In the future only where the mask is -1 will be replaced with generated tokens. 

Then we run a loop which iterates over the tokens is made to run. In the first pass, the entire prompt is fed in to the model, with starting_pos as 0. From the next iteration onwards only one token will be sent for forward pass to the model. This can be validated by checking the difference between the `prev_pos` and `cur_pos` . After the first iteration, the difference will always be 1. 

<aside>
💡 Note that starting position is passed in to the transformer because that is used later in the caching in the self attention mechanism. And, after the first iteration as the sequence lenght will always be 1, there is no need for an attention mask.

</aside>

As we know already, no matter the input size, the output will always be the logits of shape (1,32_000). Softmax is applied on them to convert them into probabilites and finally, one token is sampled out of the logits (considering it a multinomial distribution). As a new token is generated, it is added to the `tokens` variable depending on the input mask, untill the tensor of shape 264 is completely filled. The loop might break in between if the model generates a token or a word that should not have been generated. `_should_stop()` takes care of this.  

After this, post processing is done on the decoded tokens so that the output is more presentable and readable to the User . Lets look at 3 very important concepts here - **Temparature** and **Sampling from the top probabilites** and the **post processing function**

**The role of temparature** 

The temperature determines how greedy the generative model is.

If the temperature is low, the probabilities to sample other but the class with the highest log probability will be small, and the model will probably output the most correct text, but rather boring, with small variation.

If the temperature is high, the model can output, with rather high probability, other words than those with the highest probability. The generated text will be more diverse, but there is a higher possibility of grammar mistakes and generation of nonsense.

![temparature](https://i.imgur.com/HCS0swK.png)

The difference between the low-temperature case (left) and the high-temperature (right) case for the categorical distribution is illustrated in the picture above, where the heights of the bars correspond to probabilities.

**Sampling from top p probabilites**

```python
def sample_top_p(probs, # a tensor of size (1, 32_000) with softmax already applied
                 p # 0.95
                 ):
    """ 
    In summary, the sample_top_p function performs top-p sampling on a given probability
    distribution. It sorts the probabilities, computes the cumulative sum, applies a 
    threshold to remove tokens with cumulative probabilities exceeding the threshold, 
    normalizes the probabilities, samples a token using multinomial sampling, and returns 
    the sampled token index.

    If the sum of probalities coming before a token is greater than p, then the token is
    not considered for sampling. This is done by setting the probability of the token to 0.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
```

The `sample_top_p` function is used to perform top-p sampling on a probability distribution. It takes two inputs:

1. `probs`: A tensor of size `(1, 32_000)` representing a probability distribution over 32,000 tokens. The softmax function has already been applied to convert the logits into probabilities.
2. `p`: A float representing the threshold probability. Tokens with cumulative probabilities above this threshold will be considered for sampling.

Here's a step-by-step explanation of the function:

1. `probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)`: The `probs` tensor is sorted in descending order along the last dimension (`dim=-1`). This sorts the probabilities from highest to lowest. `probs_sort` contains the sorted probabilities, and `probs_idx` contains the corresponding indices.
2. `probs_sum = torch.cumsum(probs_sort, dim=-1)`: The cumulative sum of the sorted probabilities is computed along the last dimension using `torch.cumsum()`. This results in a tensor `probs_sum` where each element represents the cumulative sum of probabilities up to that point.
3. `mask = probs_sum - probs_sort > p`: A boolean mask is created by comparing the difference between `probs_sum` and `probs_sort` with the threshold `p`. The mask will be `True` for tokens whose cumulative probability exceeds the threshold `p`.
4. `probs_sort[mask] = 0.0`: The probabilities corresponding to the tokens that exceed the threshold are set to `0.0`. This effectively removes those tokens from consideration during sampling.
5. `probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))`: The `probs_sort` tensor is divided by its sum along the last dimension (`dim=-1`). This normalizes the probabilities so that they sum up to 1, ensuring they form a valid probability distribution.
6. `next_token = torch.multinomial(probs_sort, num_samples=1)`: Using `torch.multinomial()`, a single token is sampled from the probability distribution defined by `probs_sort`. The `num_samples` parameter is set to `1`, indicating that only one token is sampled. 
    
    Multinomial sampling is a method used to randomly draw samples from a categorical distribution. In the context of probability distributions, a categorical distribution represents a discrete probability distribution over a finite set of categories or outcomes.
    
    When using multinomial sampling, you have a categorical distribution defined by a set of probabilities assigned to each category. The probabilities sum up to 1, representing the likelihood of each category being chosen. Multinomial sampling allows you to randomly select one or more categories based on their probabilities.
    
    Here's how multinomial sampling works in PyTorch:
    
    1. You provide a tensor containing the probabilities of each category. This tensor should have the shape **`(batch_size, num_categories)`**.
    2. Specify the **`num_samples`** parameter, which determines how many samples you want to draw for each category.
    3. PyTorch uses the probabilities to sample randomly from the categorical distribution. The number of samples drawn for each category follows a multinomial distribution, which is determined by the probabilities.
    4. The output of multinomial sampling is a tensor containing the indices of the sampled categories. The shape of this tensor is **`(batch_size, num_samples)`**.
    
    By using multinomial sampling, you can randomly select categories according to their probabilities. This technique is commonly used in various applications, including generating text, image captioning, and reinforcement learning, where random exploration is necessary.
    
7. `next_token = torch.gather(probs_idx, -1, next_token)`: The sampled token index is obtained by using `torch.gather()` to gather the corresponding index from `probs_idx`. This ensures that the sampled token index matches the original indexing of the input tensor.
8. `return next_token`: The sampled token index is returned as the output of the function.

In summary, the `sample_top_p` function performs top-p sampling on a given probability distribution. It sorts the probabilities, computes the cumulative sum, applies a threshold to remove tokens with cumulative probabilities exceeding the threshold, normalizes the probabilities, samples a token using multinomial sampling, and returns the sampled token index.

**Post Processing function** 

```python
def postprocessing(output_text, stop_words=None, threshold=10):
    
    sentences = output_text.split(".")
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > threshold and sentence[-1] == ".":
            filtered_sentences.append(sentence)
    r = '.'.join(sentences).strip()
    if stop_words:
        for w in stop_words:
            if r.endswith(w):
                r = r[0:-len(w)].strip()
    if r[-1] != '.':
        r += '...'
    return r
```

The **`postprocessing`** function takes an **`output_text`** string and performs some post-processing steps to clean and format the text. Let's break down the function step by step:

1. Splitting into sentences: The **`output_text`** string is split into sentences using the period (".") as the delimiter. This is done with the **`split()`** method, which returns a list of individual sentences.
2. Filtering sentences: Each sentence is processed individually in a loop. First, any leading or trailing whitespace is removed by using the **`strip()`** method. Then, the length of the sentence is checked against the **`threshold`** value. If the sentence is longer than the threshold and ends with a period (indicating a complete sentence), it is considered valid and added to the **`filtered_sentences`** list.
3. Joining filtered sentences: The **`filtered_sentences`** list is joined back into a single string using the period as the separator. This step is performed with the **`join()`** method, resulting in a string that contains only the valid sentences.
4. Handling stop words: If the **`stop_words`** parameter is provided, the function checks if the resulting string **`r`** ends with any of the stop words. If a stop word is found at the end of the string, it is removed by slicing the string from the beginning up to the length of the stop word. The resulting string is then stripped of any leading or trailing whitespace.
5. Ensuring sentence ending: Finally, the function checks if the resulting string **`r`** ends with a period. If it doesn't, a period and ellipsis ("...") are appended to the string.

The purpose of this post-processing function is to clean up the generated text and ensure that it contains valid and properly formatted sentences. It also provides the flexibility to handle specific cases such as removing stop words and enforcing proper sentence endings.

# Conclusion

Thats it!! I know this was a long blog. We looked at why the LLaMA models are relevant in today’s world, we dived deep into the tokenizer it uses, what architectural changes it included in the classic transformer models and what is the mechanism behind generating a new token. If stuck somewhere considering having a look at my [fork](https://github.com/nishantb06/pyllama) of Pyllama package. The code here is heavily commented and I’m sure this will help. Please reach out on twitter [@itsnishant14](https://twitter.com/itsnishant14) incase of any questions or discrepancies. 

Thanks for giving this a read :)
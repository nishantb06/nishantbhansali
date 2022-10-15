---
title: "Mobile-VIT [Paper Summary]"
date: 2022-09-09T21:30:05+05:30
draft: false
---

[Papers With Code](https://paperswithcode.com/paper/mobilevit-light-weight-general-purpose-and)

## Observations

1. Theres a **global inductive bias** in CNN’s (invariance to shift and scale) which is why CNN’s have comparable performance w.r.t Transformers (Reference to this statement is in the Transformer survey paper). Transformer models overcome this with the help of extensive training regimes, large datasets and larger models. (It will be good if we mention this in the paper somewhere)
2. CoreML library was used to perform testing on I- phone 12

## Good things about the paper

1. the paper has two significant contributions
    1. A novel architecture which **combines convolution block from MobileNetV2 and the self attention block.** This is how it captures global and local dependencies 
    2. Introduces a **Multi-scale sampler for training efficiency.** Fine tuning on images with a larger resolution is a well know method to boost training accuracy. Methods have also been known which can introduce larger resolution images during the training process itself. But these guys have written a new sampler which varies the batch size according to the size of the image. **Smaller images will have larger batch size and vice versa.** Every i’th iteration introduces a smaller batch but with a large batch size. This may not be relevant to us as we don’t plan to include a survey of train on/for edge devices, but it is good to know how to boost the training accuracy of models like these.
    
    1. They have also experimented with MobileVIT as the backbone to downstream tasks like detection and segmentation, showing results like - 
        
        > The performance of Deep-Lab-v3 is improved by 1.4%, and its size is reduced by 1.6× when MobileViT is used as a backbone instead of Mobile-Net-v2. Also, MobileViT gives competitive performance to model with ResNet-101 while requiring 9× fewer parameters; suggesting MobileViT is a powerful backbone.
        > 
    2. The architecture is simple itself. They start with a couple of Mobile-Net-v2 blocks which downsamples the input. after this a Self attention layer is used on the processed feature map (note that the input shape is the same as the output shape for these layer). This output is then concatenated with the outputs from a parallel convolution operation. Then again point-wise convolutions are used on this concatenated layer . This whole process is used twice (two transformer layers only)
    3. I really like the idea of fusing attention and convolutional outputs with the help of another convolutional layer. Do LMK if this was original idea or copied from some other paper before this one

## Bad things about the paper

1. Just introducing transformer layers at two places in the model and the calling the model “VIT” makes no sense to me. It is clear that the model is Convolutional in nature. They have themselves mentioned that the significant amount of parameters come from these 2 layers. Also theres no experiment to show that the model gets a boost in performance because of these 2 layers. For example they can **replace the Attention layer and perform a couple of experiments to show that the model doesn’t perform as good as it does with the attention layer**.
2. They said they have used the swish activation function for the entire model. Yes theoretically its better than a simple linear activation function but for an architecture to be deployed on edge devices i would rather **add more parameters that to waste computation on a complex non linear activation function**.
3. The exact value of FLOPS is not mentioned for any variant of the model. They just mention that it is roughly half the FLOPS of DeIT on image-net dataset. So we will have to get the exact value of FLOPS on our own. (How do we calculate FLOPS with code btw?)
4. Recommend everyone to go through the final paragraph of the paper labelled discussion. It says that even though the model is smaller then some well know CNN’s, on mobile devices because - 
    
    > 
    > 
    > 
    > This difference is primarily because of two reasons. First, **dedicated CUDA kernels exist for
    > transformers on GPUs**, which are used out-of-the-box in ViTs to improve their scalability and efficiency on GPUs (e.g., Shoeybi et al., 2019; Lepikhin et al., 2021). Second, CNNs benefit from several device-level optimisations, including batch normalisation fusion with convolutional layers (Jacob et al., 2018). These optimisations improve latency and memory access. However, such dedicated and optimised operations for transformers are currently not available for mobile devices
    > 
5. They haven’t used positional embedding in their transformer layers

### Fun Fact 
- Layer Norms are used in transformer models because the batch size has to be kept too small because of the large size of transformer models. Batch size has to be kept extremly low (i have myself used 2 or 4 as batch-size),and as batch_norm us not that effective when batch size is so low. We use learning rate warmup for the same reason 
# **A Self-Supervised Descriptor for Image Copy Detection - Review**

[[Paper](https://arxiv.org/abs/2202.10261)][[Code](https://github.com/facebookresearch/sscd-copy-detection)]

They have built upon the work of SimCLR and successfully tackled its limitations.
Do give this paper a read if you are looking for a way of generating powerful embeddings/descriptors for your image dataset.  

## Good things about the paper

- It Introduces regularisation term based on Entropy which is used to make the descriptors more sparse. Which means that negative images wont be as “close” to each other as they used to be in SimCLR. By doing this it overcomes a major drawback of SimCLR that a descriptor of size 128 was as efficient as a descriptor of size 512.
- It adds more robust augmentations to its augmentation pipeline. It has also adapted the InfoNCE loss function to make it suitable for cut-mix/mixup augmentation
- It makes good use of GeM Pooling which was heavily used in the Instance matching genre. Ablations studies also prove its importance
- Post Processing of descriptors is also very Innovative, whitening + L2 norm from the impressive FAISS library is something that I will have to definitely look into more

## Bad things about the paper

- They should have started with a more powerful model like ViT instead of a ResNet. Will have to check their code if it can be modified for “plug n play” for different models. Encoders for transformers should be modified to give suitable descriptors as Encoders are known for their pre-training capabilities and can be used for larger images as well
- Maybe SWaV could also be an inspiration to build upon this paper as it uses clustering algorithms to provide training time labels. For a task of copy detection this looks like a suitable task
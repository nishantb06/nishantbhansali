---
title: "Portfolio"
date: 2024-05-04T21:25:02+01:00
draft: false
menu: "main"
---

Hi, after my extensive work on Content Moderation Systems at Sharechat, I consider myself a generalist who can train, finetune, deploy Deep Learning Models across different modalites (Vision, NLP, Audio) in scalable production environments. My everyday work has included everything from Microsoft Excel to Kubernetes and working together with folks from Product, Operation, Data Scientists and Engineers alike. With a base in Computer Vision, I have a thorough understanding of how to train/finetune LLM's and how to build RAG applications with them as well.

My team at Sharechat consists of 3 people for its two products, Moj and Sharechat. As Moj is a bigger, more complex product, my 2 colleages take ownership of it. I have been taking ownership of every ML pipeline in the Sharechat ecosystem.This involves trainiing,deploying and monitoring pipelines. We often work together to share data, come up with new solutions, share code and act as on-calls for each other! 


## ML Engineer @ Sharechat (June 2022 -> Present)
### Notable Projects 

#### ML Server optimisation for Moj Audio Livestream moderation
There is a need to check for NSFW content during Livestream on MOJ. For this we trained Wave2Vec2 models on 20 Second Audio Clips for different languages. These models were earlier deployed in an event driven architecture applications written in Python. 

Rewriting this application in Golang and leveraging Torchserve (Scripting the model + efficent process management) to expose the models through an API , helped us cut the cost of this deployment by 66%, yielding in savings of 6L rupees per month. Cost of the infrastructure used to host this application went down from 9L per month to 3L per month.

Torchserve was chosen because it can deploy the same model in multiple processes , all the while exposing it as an api. it's backend (written in Java) can efficiently manage these processes, for example if one of them shuts down, it will automatically spin up a new process to replace the old one. It also has great support for logging of system and ML Model metrics via Prometheus and Grafana. Torchserve can efficiently run in the background, with the Golang code being the main interface for the application.A custom docker container which had Go, Python, Torchserve and all the necessary packages for inferncing installed in it. This container was used to deployed to a Kubernetes cluster

Tracing the model also gave a significant speedup's as tracing is about capturing the operations of a model to create a script representation for efficient inference

Golang was chosen to leverage its intuitive syntax for concurreny. Goroutines and channels make it extremely easy to manage concurrent threads. Fan in Fan out design pattern was used to process a batch of 8 (or less) messages together. Updates to databases, Pushing different events to downstream queue's was done by a simple Fire and Forget go routine as there is no dependency of these operations with the main flow. 
#### Min View Tool : 
This is one of the projects where I like to point out that training an ML model is not needed everywhere. Efficient data analysis can bring you the desired results as well. 

A post in its lifecycle gets moderated twice, once whne it is created and as a safety net, every post in Sharechat that receives 1000 views gets moderated again. At the first line of defence (when the post gets created), a ML Model give the post a score between 0 and 1 , representing the likelihood of the post being NSFW.

With a simple analysis we recognised that if we do not moderate those posts with NSFW score less than a particular threshold at this 1k views safety net, we wouldn't be missing a significant number of posts. The threshold was set such that the recall of this simple heuristic based model will be 95%. 

And guess what, we created another safety net at 2000 views to catch this missed 5% recall. This method helped us to reduce to volume of posts to be manually moderated, thus leading to less operational costs (6.4L per month)

 We also tried to train a model based on realtime embeddings of the post from the recommender system model. But till 1k Views these embedding dont mature enough, which is why we did not get a high accuracy here. 

Another thing which pushed the accuracy of this model here is leveraging the past history of the user. We further trained a tree based model with 2 user features  number of posts created by the user in the last 6 months, and the number of discards for this user in the last 6 months. we set up a service to fetch this data from the corresponding database. 

#### CLIP integration for VRT :
VRT or the video review tool is the first line of defence where every new post that gets created, is assigned a score between 0 and 1, representing the likelihood of the post being NSFW. This has to be done within 10 minutes of the post being created. 

To get this score, we use a Pre trained deep learning models as a feature extractors for audio video and text present in the video posts. 
For audio we use an in house VGGish based model which gives us an embedding of shape 128
For text we use an in house XLMR based model which gives embedding of shape 1024
For Visual Features we use CLIP model which gives an embedding of shape 2048. For video posts we extract out the frames at a second's interval, then average out their embeddings to get the final embedding for the video.

These 3 embeddings are then concatenated and a neural network is trained using these concatenated features as input. The NN is trained in a multiclass classification setting, where we try to classify the image input into 7 categories.  These 7 categories are something like Violence, Nudity, Abusive etc. Basically the bifurcations in the NSFW data that we see.

#### 


## Computer Vision and Image Processing Intern @ Sharechat (Jan 2022 -> May 2022)

- Implemented various research papers for Contrast Enhancement and Denoising like Contrast Limited Adaptive Histogram Equalization and Non-Local Means denoising and other classical Image Processing algorithms

- Executed thorough benchmarking analyses on proprietary datasets,tech stack comprising C++, Python libraries including OpenCV and scikit-learn, along with other open-source repositories.

## Personal Projects
### 3D Ray tracing Engine
**A 3D ray tracing engine written from scratch in C**

### Emblio
**Package for generating embeddings of different content types**


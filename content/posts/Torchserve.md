---
title: "Deploying Models at Scale using Torchserve"
date: 2024-03-31T23:39:16+05:30
draft: true
---

- AI Shit
    
    TorchServe is a flexible and easy to use tool for serving PyTorch machine learning (ML) models at scale. It is part of the PyTorch ecosystem and was developed in collaboration with AWS to facilitate the deployment of PyTorch models in production environments.
    
    TorchServe simplifies the process of deploying PyTorch models by providing a straightforward and standardized way to package and serve them. It supports multiple types of models, including those for image and text classification, object detection, and more.
    
    TorchServe includes features that are essential for deploying models at scale, such as multi-model serving, model versioning for A/B testing, metrics for monitoring, and RESTful endpoints for application integration.
    
    It's also designed to be highly customizable, allowing developers to use custom handlers and pre-processors. This means that TorchServe can adapt to various deployment scenarios and cater to different needs.
    
    In summary, TorchServe is a powerful tool for deploying ML models at scale. It provides a robust, flexible, and easy-to-use solution for serving PyTorch models, making it an excellent choice for production environments.
    
    Continuing from where we left off, one of the main advantages of TorchServe is its ability to handle large and diverse workloads. It can serve multiple models simultaneously, which makes it ideal for applications that require a variety of ML models to operate in real-time. Additionally, it can manage heavy traffic, maintaining performance under high loads.
    
    TorchServe also integrates seamlessly with AWS services, enabling developers to leverage the scalability and flexibility of the cloud. This means you can easily deploy your models on AWS, automate scaling based on demand, and take advantage of other AWS features to enhance your deployment.
    
    Moreover, TorchServe's support for model versioning is a significant feature. It allows for A/B testing of different model versions to determine the most effective one. This is crucial for continuous improvement and optimizing the effectiveness of your ML models.
    
    Lastly, TorchServe provides robust metrics for monitoring your models. This allows you to track the performance of your models in real-time, so you can quickly identify and address any issues that arise.
    
    In conclusion, TorchServe is not just a tool for serving PyTorch models, but a comprehensive solution for deploying ML models at scale. Its combination of flexibility, scalability, and robust features make it a game-changer in the deployment of ML models.
    

Imagine an AI application which needs to serve Millions of requests in a day. In the backend you have multiple ML models on which you need to perform inferencing , and the number of requests each model receives keeps changing throughout the day. Some models need a GPU to run on whereas other models are smaller and can run on the CPU only. Not just that , over a period of time , these models will need to be retrained and newer versions will have to be deployed. How do you manage the versions and dynamically swap out the old models?

Torchserve is one such open source tool, that can help you manage , speed up and monitor your Machine Learning Models in production! 

The Problems that Torchserve tackles are:

1. Torchserve exposes your Model as an endpoint. This makes other inferencing much simpler for the backend servers! Getting the scores or the class in which it belongs to is just one HTTP request away :). 
2. The same model is loaded into multiple processes and these processes are managed by Torchserve’s backend (which is written in Java). New processes are created and killed on the fly as the application continues to do it’s job in a production environment
3. Torchserve provides functionality for batching, this means higher throughput and efficient use of costly GPU’s. This functionality can be customised according to your requirements.  For example, after setting the batch size as 8 and a waitTime as 10 seconds, torchserve will wait for 10 seconds for the batch to fill up and then perform inferencing collectively on the batch. If after 10 seconds the batch is lets say filled only with 6 requests , it will perform inferencing with that batch size only. 
The best part is the batch inferencing is managed by torchserve , which means you don’t write code to prepare a batch in your backend application first and therefore the endpoint remains the same as well!
4. Torchserve provides a good set of management API’s, which allows you to change multiple configurations like BatchSize and number of workers on the fly. Along with logging, this functionaluty can be used to manage your Inference server even better. Logging ofcourse integrates with Grafana and Prometheus.
5. 

### Links

1. [Torchserve Docs](https://pytorch.org/serve/) 
2.
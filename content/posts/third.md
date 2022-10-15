---
title: "Docker Cheatsheet"
date: 2022-09-11T10:15:51+05:30
draft: false
---

# But it works on my machine !!??
The above sentence is exactly the problem docker solves - 
- Earlier there was no way to run 2 applications (different OS) on the same machine. VMware solved this problem by introducing Virtual Machines. But we would have to separately assign RAM and storage for our second machine. This was still a bottleneck as we can't ship applications effectively with this, which is why Docker was invented. Docker is used so that we can share applications, including the operating system it was built on along with all of the dependencies of the applications. 

## Definitions
- **Image** : An image is a template of our source code along with all dependencies
- **Container** : A container is a running instance of an image, if an image is equivalent to a class then a container is equivalent to an object of that class
- **Dockerfile** : It is used to create the Docker Image
- **Docker Registry** : Somewhere people can push their images where it can be accessed publicly. The most famous is DockerHub
- **Docker Daemon** : Docker runs on a client-server architecture system. Docker Daemon is the server side of it
- **Docker Runtime** : 
- **Docker Engine** : 

## Commands : 
1. start a container from ana already downloaded/built image

    -it is for the interactive shell
    ```shell
    docker run -it <IMAGE_NAME>/<IMAGE ID>
    ```
2. list all running containers
    ```shell
    docker ps
    ```
3.  Build an image from a dockerfile 
    Dockerfile should be present in the current working directorys
    ```shell
    docker build -t <IMAGE_NAME> .
    ```
    Always keep your IMAGE_NAMe as username/<image_name>, so that it is easier to push to dockerhub
4. These commands are used to list all images downloaded in the local. It also lists what are the size of these images and their pseudo-names along with their hash-ids
```shell
docker images
docker images ls
```
## How to make a DockerFile for ML projects
A Dockerfile is used to build a docker image which clones the contents of our project, sets up the base Operating systems and also downloads all the dependencies.
1. Create a file named Dockerfile and .Dockerignore file in the root of the project
2. Choose a base image, usually for python projects it is the Slim or the alpine Version taken from DockerHub. The `FROM` command in Dockerfile does just that. 
3. Then set the Working directory in your image and give it a name.
4. Copy the requirements.txt file to the working directory of the image
5. Install the dependencies via pip install -r requirements.txt. Also delete the cache of pip install to reduce the size of the image
6. Then copy all the project files. Note that project files are copied only after installing the packages because if we make a change to the code then we dont necessarily want to compile the layer of the Docker Image which deals with all the dependencies but if make any changes to the requirements then we definitely want to built the layer again where we compile all the source files. This is a Docker standard best practice when dealing with Docker Containers.

## Docker Errors and how to solve them
1. 
    ```shell
    Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
    ERRO[0000] Can't add file /Users/nishantbhansali/Desktop/image-quality-assessment/.git/hooks/update.sample to tar: io: read/write on closed pipe 
    ERRO[0000] Can't close tar writer: io: read/write on closed pipe
    ```
    Simply by restarting docker desktop. Maybe it looses connection to docker daemon when mac restarts

### How to mount a docker volumne

mount the current volume to the desired location in the image. Here we mount it to /workspace/project.
-v or --volume tags is used for this

remember that the mount is two ways, if we change any file in the docker image, it reflects in the local as well.


```shell
docker run -it -v $(pwd):/workspace/project 63b0afa6efdc bash
```


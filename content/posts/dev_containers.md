---
title: "Dev Containers: Open Vscode editor inside any Docker image"
date: 2024-01-01T14:10:41+05:30
draft: true
---

Let’s say you have a docker image for an application and you want to run some test or experiment/add some new feature to that application. Normally I would build that image locally and mount the application directory as volume when I run that container. But something better exists

Using Dev Containers is better because 

1. It gives the VS code experience for any docker image. 
2. Different VS code extensions can be used here like Linting, Copilot etc.
3. Allows reproducabilty for any other developer. 

[Video](https://www.youtube.com/watch?v=b1RavPr_878)

# How to create a dev container

### Step 1: Create config files

Open the command pallete in VS Code by `CMD + Shift + p` or `Cntrl + Shft + p` if you are in Windows. type this `Dev Containers: Add dev container config files` . Choose the language you want and any other functionality you want and press enter.

This will create a `devcontainer.json` file inside a `.devcontainer` folder in your root

this file will look something like this

```json
// For format details, see https://aka.ms/devcontainer.json. For config options, see the
// README at: https://github.com/devcontainers/templates/tree/main/src/go
{
	"name": "Go",
	// Or use a Dockerfile or Docker Compose file. More info: https://containers.dev/guide/dockerfile
	"image": "mcr.microsoft.com/devcontainers/go:1-${templateOption:imageVariant}"

	// Features to add to the dev container. More info: https://containers.dev/features.
	// "features": {},

	// Use 'forwardPorts' to make a list of ports inside the container available locally.
	// "forwardPorts": [],

	// Use 'postCreateCommand' to run commands after the container is created.
	// "postCreateCommand": "go version",

	// Configure tool-specific properties.
	// "customizations": {},

	// Uncomment to connect as root instead. More info: https://aka.ms/dev-containers-non-root.
	// "remoteUser": "root"
}
```

Here replace the name field to whatever is suitable and replace the image field by the name of your own custome image. Can see possible names by running `docker images` in you CLI

<aside>
💡 the image provided here is of native Golang, pulled from microsoft container registry

</aside>

Finally it should look something like this

```json
// For format details, see https://aka.ms/devcontainer.json. For config options, see the
// README at: https://github.com/devcontainers/templates/tree/main/src/go
{
	"name": "Go-dummy-app",
	// Or use a Dockerfile or Docker Compose file. More info: https://containers.dev/guide/dockerfile
	// "image": "mcr.microsoft.com/devcontainers/go:1-1.21-bullseye",
	"image": "dummy-app"

	// Features to add to the dev container. More info: https://containers.dev/features.
	// "features": {},

	// Use 'forwardPorts' to make a list of ports inside the container available locally.
	// "forwardPorts": [],

	// Use 'postCreateCommand' to run commands after the container is created.
	// "postCreateCommand": "go version",

	// Configure tool-specific properties.
	// "customizations": {},
	

	// Uncomment to connect as root instead. More info: https://aka.ms/dev-containers-non-root.
	// "remoteUser": "root"
}
```

### Step 2 : Open dev container

Again open Command pallete and type `Dev Container: reopen in container`

This will take a bit to reload and viola! you should be ready to go now

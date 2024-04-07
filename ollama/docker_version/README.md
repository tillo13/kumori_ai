# How to Install and Run the GEMMA Model Locally

This guide will show you how to install and run Google's GEMMA model using Ollama and Docker on your local machine.

## Requirements

1. Ollama version 0.1.26 or higher. You can download it from the [Ollama's Docker Hub](https://hub.docker.com/r/ollama/ollama/tags) or the official [Ollama website](https://ollama.com).
2. Docker
3. Docker Compose
4. The Docker Compose file from the [Open WebUI GitHub repository](https://github.com/open-webui/open-webui). If you are using a general-purpose computer download `docker-compose.yaml`; if using a machine with an NVIDIA or advanced processor, download `docker-compose.gpu.yaml`.

## Steps

1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Download the appropriate Docker Compose file from the [Open WebUI GitHub repository](https://github.com/open-webui/open-webui) and navigate to the directory containing your Docker Compose file in the terminal.
3. Pull the latest Ollama Docker image using the command `docker pull ollama/ollama:latest`.
4. Build and run the Docker Compose file using the command `docker-compose -f <docker-compose-file> up -d`. Replace `<docker-compose-file>` with your selected Docker compose file. The `-d` flag runs it in the background. If the name "/ollama" is already in use by another container, you might need to remove or rename it before running this command.  Example: docker-compose -f docker-compose.yaml up -d
5. Now you should be able to visit [http://localhost:3000/auth](http://localhost:3000/auth) to sign up or log in (it's local so feel free to create any username/pw you want).
6. In Open WebUI, go to Settings → Models. In the field "Pull a model from ollama.com", enter the exact model name from [Ollama library](https://ollama.com/library). For GEMMA, use `gemma:2b` or `gemma:7b` depending on the level of resources you want to dedicate to the model.
7. After successfully adding the model, go back to the interface and click on "Select Model" to select GEMMA. GEMMA should now run in your Docker container.
8. You should now be able to chat with GEMMA through the chat interface in the web UI.

## Uninstalling GEMMA and Related Infrastructure

If you want to completely remove the installed GEMMA model and all related infrastructure, you can follow the steps below. Please note, this will not only remove the GEMMA model but also all related data and Docker configurations.

## Steps

1. **Stop the Running Docker Containers:** Before removing anything, you'll first need to stop all running Docker containers. This can be done using the command: `docker-compose down`.

2. **Remove Docker Images:** The next step is to remove the Docker images related to the GEMMA project. You can use the command `docker rmi $(docker images -q)` to remove all Docker images, or replace `$(docker images -q)` with the specific image ID you wish to delete if you want to remove specific images.

3. **Remove Docker Volumes:** Docker volumes hold the data related to running Docker containers. To delete them, you can use the command `docker volume rm $(docker volume ls -q)`. This will delete all Docker volumes. Replace `$(docker volume ls -q)` with the specific volume ID to delete a specific volume.

4. **Delete the Project Directory:** Finally, delete the project directory from your file system. This will permanently delete all the project files, so make sure you have a backup of any important files.

## To Shutdown and Restart GEMMA Docker Containers

For shutting down (stopping and removing) the Docker containers without deleting any data or configurations:
1. Open your terminal or command prompt.
2. Navigate to the directory where your Docker Compose file is located. You can use `cd path/to/directory`.
3. Run the command `docker-compose down`.

To restart (once shutdown):
1. Open your terminal or command prompt.
2. Navigate to the directory where your Docker Compose file is located. You can use `cd path/to/directory`.
3. Run the command `docker-compose up -d`.

The `-d` flag in the `docker-compose up -d` command makes the services run in the background.

## Screenshots

![Alt text](https://github.com/tillo13/kumori_ai/blob/main/ollama/docker_version/screenshots/0_download_openai.png "Screenshot 1")
![Alt text](https://github.com/tillo13/kumori_ai/blob/main/ollama/docker_version/screenshots/1_download_docker.png "Screenshot 2")
![Alt text](https://github.com/tillo13/kumori_ai/blob/main/ollama/docker_version/screenshots/2_docker_values.png "Screenshot 3")
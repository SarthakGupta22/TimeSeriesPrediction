# Running the Dockerized Container

Welcome to the Dockerized version of Receipt Counter! 
This guide will help you set up and run the Docker container containing the application.
The full code is also available on Github.

## Prerequisites

Make sure you have the following prerequisites installed on your system:

- [Docker](https://www.docker.com/get-started)

## Steps

### 1. Import the Docker Image
#### Docker image can be downloaded [here](https://drive.google.com/file/d/18Ywm2AmNO4FkvOVVH78G5KyGV_tQk2M9/view?usp=sharing)

```bash
docker load -i receipt_counter.tar
```

### 2. Verify the Imported Image

```bash
docker images
```

### 3. Run the docker container

```bash
docker run -p 5000:5000 receipt_counter
```

#### Replace 5000:5000 with the appropriate port mapping if your application uses a different port.

### 4. Access the application

#### Navigate to [http://localhost:5000](http://localhost:5000) to access the web application.

## Additional Considerations

### Operating System Compatibility:
#### Ensure that you are running a compatible operating system (Windows, macOS, Linux) with Docker installed.

### Port Conflicts:
#### If there are port conflicts with other applications on your system, adjust the port mapping when running the container.

### Firewall and Security Settings:
#### Ensure that your firewall and security settings allow incoming and outgoing traffic on the specified port.

## Cleanup(Optional)

### If you no longer need the container, you can stop and remove it using the following commands:

```bash
docker ps
docker stop <container_id_or_name>
docker rm <container_id_or_name>
```


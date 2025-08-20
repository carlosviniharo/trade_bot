# trade_bot
This is a microservice to extract high changes in the volume of the coins listed on Binance Futures.

## How to create the mongo db container.
1. Run the following command to create the mongo db container.

```bash
docker run -d -p 27017:27017 --name mongo mongo
```

## How to run the project in your local docker environment.
1. Clone the repository.
2. Run the following command to build the docker image.
```bash
docker build -t trade_bot .
docker run -v $(pwd):/app -p 8000:8000 trade_bot
docker run -v ${PWD}:/app -p 8000:8000 trade_bot 
```
Note: Do not include the library TaLib in the requirements.txt file. It is already included in the Dockerfile.
You can use this command to debug the docker in case of any issues.

```
docker run -it --entrypoint /bin/bash trade_bot
```
### Kubernetes Deployment Documentation

### Navigate to the k8s directory of the project.
```cd k8s``` 

### Direction of the registry of the docker image.
```us-central1-docker.pkg.dev/inspired-oath-441023-v1/docker-repo/ ```

### Please include How to deploy the project in Google Cloud Run.
```gcloud container clusters get-credentials my-first-cluster-2 --zone northamerica-northeast2-a --project inspired-oath-441023-v1```

### Notice that you should set the .env file in the k8s directory of the project.
```bash
kubectl create secret generic tradebot-env --from-env-file=.env
```

### For the manual deployment of the projects in Google Cloud Run from the k8s directory.
```bash
kubectl apply -f .
```

# trade_bot
This is a microservice to extract high changes in the volume of the coins listed on Binance Futures.

## How to create the mongo db container.
1. Run the following command to create the mongo db container.

```docker run -d -p 27017:27017 --name mongo mongo```

## How to run the project in you local docker environment.
1. Clone the repository.
2. Run the following command to build the docker image.

```docker build -t trade_bot .```
```docker run -v $(pwd):/app -p 8000:8000 trade_bot```
```docker run -v ${PWD}:/app -p 8000:8000 trade_bot```

Note: Do not include the library TaLib in the requirements.txt file. It is already included in the Dockerfile.
You can use this command to debug the docker in case of any issues.

```docker run -it --entrypoint /bin/bash trade_bot```



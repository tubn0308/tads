## install docker
get docker desktop here and install it.
https://docs.docker.com/docker-for-mac/

## create docker image and container
open terminal and enter the following command.

```
docker-compose up -d --build
```

## enable docker container
```
docker exec -it tads bash
```
## run flask app
run flask server.
```
cd tads/app
python views.py
```
visit http://localhost:3939 on your browser.

## stop container
```
Ctrl + p, q (hold Ctrl down and press p and q sequentially)
docker stop tads
```

## start container again
```
docker start tads
```
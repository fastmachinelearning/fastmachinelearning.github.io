# Instructions for testing the website locally 

## Conda

```
source create-local-conda-env.sh 
jekyll serve --incremental
```
Then open the page in your browser!

## Docker

```
docker build -f Dockerfile-local-mac -t my-jekyll .
docker run -d -p 8080:80 my-jekyll
docker ps
```

Open: http://localhost:8080/




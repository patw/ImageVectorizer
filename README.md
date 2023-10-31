# ImageVectorizer
A python FastAPI vector service for generating dense vectors from text and images using the CLIP model.

The service will let you generate 512 dimension vectors for text or images for image similarity and image text search.

Does not need GPU to run.

## Local Installation

```
pip install -r requirements.txt
```

## Local Running

```
uvicorn main:app --host 0.0.0.0 --port 3001 --reload
```
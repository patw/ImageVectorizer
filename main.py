from fastapi import FastAPI, UploadFile
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from PIL import Image
from sklearn.preprocessing import normalize
import numpy
from scipy.spatial.distance import euclidean
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import requests
from io import BytesIO
import json

# Load in resnet image labeling model
resnet = fasterrcnn_resnet50_fpn(pretrained=True)
resnet.eval()

# Load COCO labels mapping for resnet50 to do image labeling
with open('cocolabels.json', 'r') as f:
    coco_labels = json.load(f)

app = FastAPI()

# Fast API init
app = FastAPI(
        title="ImageVectorizor",
        description="Create dense vectors for images and text using CLIP (clip-vit-base-patch32). Use the similarity tools to compare images and text.",
        version="1.0",
        contact={
            "name": "Pat Wendorf",
            "email": "pat.wendorf@mongodb.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/license/mit/",
    }
)

# Load the more recent CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name)

# Perform cosine similarity check between two vectors and output the measurement (thanks VectorService!)
def similarity(v1, v2):
    # Define two dense vectors as NumPy arrays
    vector1 = numpy.array(v1)
    vector2 = numpy.array(v2)

    # Compute Euclidean distance
    euclidean_distance = euclidean(vector1, vector2)

    # Compute dot product
    dot_product = numpy.dot(vector1, vector2)

    # Compute cosine similarity
    cosine_similarity = numpy.dot(vector1, vector2) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))

    return {"euclidean": euclidean_distance, "dotProduct": dot_product, "cosine": cosine_similarity}

# L2 normalize a tensor
def l2_norm(tensor_data):
    l2_norm = torch.norm(tensor_data, p=2)
    return tensor_data / l2_norm

# Extracts image labels from a PIL image using resnet50 and the COCO labels
def get_image_labels(image):
    # Convert image to tensor first
    image = F.to_tensor(image)
    image = image.unsqueeze(0)

    # Create the predictions on the tensor
    with torch.no_grad():
        predictions = resnet(image)
    labels = predictions[0]['labels'].tolist()

    # Match the COCO labels with the labels list
    captions = []
    for label in labels:
        captions.append(coco_labels[str(label)])
    return list(set(captions))

# CLIP Text Embedding
def get_clip_text_embedding(text):
   inputs = tokenizer(text, return_tensors = "pt")
   text_embeddings = model.get_text_features(**inputs)
   norm_embeddings = l2_norm(text_embeddings)
   embedding = norm_embeddings.cpu().detach().numpy()
   return embedding.tolist()[0]

# CLIP Image Embedding
def get_clip_image_embedding(upload_image):
   image = processor(text = None,images = upload_image, return_tensors="pt")["pixel_values"]
   image_embedding = model.get_image_features(image)
   norm_embeddings = l2_norm(image_embedding)
   embedding = norm_embeddings.cpu().detach().numpy()
   return embedding.tolist()[0]

# Endpoint to get text embedding
@app.post("/text_vector")
async def text_vector(text: str):
    return get_clip_text_embedding(text)

# Endpoint to upload image and return vector
@app.post("/upload_image_vector")
async def upload_image_vector(image: UploadFile):
    image = Image.open(image.file)
    return get_clip_image_embedding(image)

# Endpoint to download image by URL and return vector
@app.post("/url_image_vector")
async def url_image_vector(image_url: str):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return get_clip_image_embedding(image)

# Endpoint to get similarity score between image and text
@app.post("/text_upload_image_similarity")
async def text_upload_image_similarity(text: str, image: UploadFile):
    image = Image.open(image.file)
    v1 = get_clip_text_embedding(text)
    v2 = get_clip_image_embedding(image)
    return similarity(v1,v2)

# Endpoint to get similarity score between text and text
@app.post("/text_text_similarity")
async def text_text_similarity(text1: str, text2: str):
    v1 = get_clip_text_embedding(text1)
    v2 = get_clip_text_embedding(text2)
    return similarity(v1,v2)

# Endpoint to get similarity score between image and another image
@app.post("/upload_image_image_similarity")
async def upload_image_image_similarity(image1: UploadFile, image2: UploadFile):
    i1 = Image.open(image1.file)
    i2 = Image.open(image2.file)
    v1 = get_clip_image_embedding(i1)
    v2 = get_clip_image_embedding(i2)
    return similarity(v1,v2)

# Endpoint to compare 2 images by URL and show similarity
@app.post("/url_image_image_similarity")
async def url_image_image_similarity(image_url1: str, image_url2: str):
    r1 = requests.get(image_url1)
    r2 = requests.get(image_url2)
    i1 = Image.open(BytesIO(r1.content))
    i2 = Image.open(BytesIO(r2.content))
    v1 = get_clip_image_embedding(i1)
    v2 = get_clip_image_embedding(i2)
    return similarity(v1,v2)

# Endpoint to get similarity score between image and text
@app.post("/text_url_image_similarity")
async def text_upload_image_similarity(text: str, image_url: str):
    request = requests.get(image_url)
    image = Image.open(BytesIO(request.content))
    v1 = get_clip_text_embedding(text)
    v2 = get_clip_image_embedding(image)
    return similarity(v1,v2)

# Endpoint for image labeling of uploaded images
@app.post("/upload_image_labels")
async def upload_image_labels(image: UploadFile):
    image = Image.open(image.file)
    return get_image_labels(image)

# Endpoint for image labeling of URL images
@app.post("/url_image_labels")
async def upload_image_labels(image_url: str):
    request = requests.get(image_url)
    image = Image.open(BytesIO(request.content))
    return get_image_labels(image)
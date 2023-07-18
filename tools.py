from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection 
from PIL import Image
from transformers import pipeline
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
import openai
from dotenv import dotenv_values

### parameters #########
config = dotenv_values('.env')
openai.api_key = config['OPENAI_API_KEY']
SECURE_CONNECT_BUNDLE_PATH = config['SECURE_CONNECT_BUNDLE_PATH']
ASTRA_CLIENT_ID = config['ASTRA_CLIENT_ID']
ASTRA_CLIENT_SECRET = config['ASTRA_CLIENT_SECRET']

cloud_config = {
    'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
    }
auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()


### Image caption tool #########
class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a detailed caption describing the image "

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # cuda

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

### Object Detection tool #########
class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

### Similarity Search tool #########
class Similarity(BaseTool):
    name = "Similarity"
    description = "Use this tool when given the path to an image that you find similarities in the vector database " \
                    "The result will be shown as images "

    def _run(self, img_path):
        KEYSPACE_NAME = 'vector'
        TABLE_NAME = 't_images' 
        model = SentenceTransformer('clip-ViT-B-32')
        img_emb1 = model.encode(Image.open(img_path))
        image_data = [(1, img_path,img_emb1.tolist())]
        
        for image in image_data:
            session.execute(f"INSERT INTO {KEYSPACE_NAME}.{TABLE_NAME} (id,name,item_vector) VALUES {image}")
        
        for row in session.execute(f"SELECT id,name, item_vector FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY item_vector ANN OF {img_emb1.tolist()} LIMIT 3"):
            if row.id != 1:
               res = row.name 
               # display image
               st.image(config['image_inputdir']+res, use_column_width=True)

        return res 

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

### Character Recognition and Vector Search tool #########
class OCRSearch(BaseTool):
    name = "OCRSearch"
    description = "Use this tool when given the path to an image that you are asked more information about a product " \
                    "The outcomes will be presented in the form of sentences."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')
        nlp = pipeline("document-question-answering", model="impira/layoutlm-document-qa",)  #a better OCR model can be used.
        res = nlp(image,"What is written in the image?")  
        KEYSPACE_NAME = 'vector'
        TABLE_NAME = 'ret_table'
        model_id = "text-embedding-ada-002"
        embedding = openai.Embedding.create(input=str(res), model=model_id)['data'][0]['embedding']
        for row in session.execute(f"SELECT document_id,document,embedding_vector FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY embedding_vector ANN OF {embedding} LIMIT 1"):
                ret_res = row.document 

        return ret_res 

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

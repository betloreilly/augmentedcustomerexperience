import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from dotenv import dotenv_values
from PIL import Image
from sentence_transformers import SentenceTransformer
 

### parameters #########
config = dotenv_values('.env')
SECURE_CONNECT_BUNDLE_PATH = config['SECURE_CONNECT_BUNDLE_PATH']
ASTRA_CLIENT_ID = config['ASTRA_CLIENT_ID']
ASTRA_CLIENT_SECRET = config['ASTRA_CLIENT_SECRET']

cloud_config = {
    'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
    }
auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
KEYSPACE_NAME = 'vector'
TABLE_NAME = 't_images2'
 
model = SentenceTransformer('clip-ViT-B-32')

location = '/Users/betuloreilly/llmdemos/augmentedcustomerexperience/images/'
img_emb1 = model.encode(Image.open(location+'zara1.jpg'))
img_emb2 = model.encode(Image.open(location+'zara2.jpg'))
img_emb3 = model.encode(Image.open(location+'zara3.jpg'))
img_emb4 = model.encode(Image.open(location+'zara4.jpeg'))
img_emb5 = model.encode(Image.open(location+'zara5.jpeg'))
img_emb6 = model.encode(Image.open(location+'zara6.jpeg'))
img_emb7 = model.encode(Image.open(location+'zara7.jpg'))

image_data = [
    (5, 'zara1.jpg',   img_emb1.tolist()),
    (6, 'zara2.jpg',   img_emb2.tolist()),
    (7, 'zara3.jpg',   img_emb3.tolist()),
    (8, 'zara4.jpeg',  img_emb4.tolist()),
    (9, 'zara5.jpeg',  img_emb1.tolist()),
    (10, 'zara6.jpeg', img_emb2.tolist()),
    (11, 'zara7.jpg',  img_emb3.tolist())
]

for image in image_data:
   session.execute(f"INSERT INTO {KEYSPACE_NAME}.{TABLE_NAME} (id,name,item_vector) VALUES {image}")
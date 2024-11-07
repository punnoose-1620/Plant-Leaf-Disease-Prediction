import tensorflow as tf
import cv2
import numpy as np
import pickle
from functools import lru_cache
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware



# model path address
MODEL_PATH = './model/EfficientNetB3-Plant Disease-98.40.h5'

# size of the image to which it would be rescale
IMG_SIZE = (224, 224)


# load the tensoflow model and cache it such that upon restarting
# it would load from the memory cache and not from the disk to make the loading process super-fast
@lru_cache()
def get_model(model_path: str):
    model = tf.keras.models.load_model(model_path)
    return model
    
    
# reading the image if the path given
def img_loading(img_path: str):
    img = cv2.imread(img_path)
    return img


# preprocess the image
def img_processing(np_img: np.ndarray):
    np_img = cv2.resize(np_img, IMG_SIZE)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    return np_img


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#app.add_middleware(HTTPSRedirectMiddleware)

print("model loading started ...\n")
MODEL = get_model(MODEL_PATH)
print("model loading completed ...\n")

# get the saved class to index mapper and vice-versa
with open('./data/class_index_dict.pickle', 'rb') as handle:
    payload = pickle.load(handle)
    class2idx = payload["class2idx"]
    idx2class = payload["idx2class"]


# get the scraped data about the disease that is predicted
# the whole information about every diseases was manually collecteed and saved as dictionary
with open('./data/disease_info.pickle', 'rb') as handle:
    disease_info = pickle.load(handle)


@app.get('/')
async def index():
    return "Welcome to Home page!"


# this api endpoint receive an image file for disease prediction
# It's convert it into numpy array, do processing and predict the class label.
# return the predicted diseases[target class] along with the follwing information:
#     -> overview about the dieases predicted
#     -> control measures that can be taken 
#     -> further links to read about more 
@app.post("/predict-from-file/")
async def create_upload_file(image_file: UploadFile = File(..., description="Upload an image file")):
    bytes_img = await image_file.read()
    np_img = np.fromstring(bytes_img, np.uint8)
    img = cv2.imdecode(np_img, 1)
    print(type(np_img), img.shape)
    img = img_processing(img)
    print(img.shape, img.max(), img.min())
    prediction = MODEL.predict(img.reshape((1, *img.shape)))
    classid_prediction = prediction.argmax(axis=1).item()
    class_predicted = idx2class[classid_prediction]
    return {"prediction_class": class_predicted, "prediction_info": disease_info[classid_prediction]}


# do the same thing as the above function except it receive path of the image
# and then it will load the file and do the rest of the things
@app.post('/predict-from-path/')
async def predict(img_path: str):
    img = img_loading(img_path)
    img = img_processing(img)
    prediction = MODEL.predict(img.reshape((1, *img.shape)))
    classid_prediction = prediction.argmax(axis=1).item()
    class_predicted = idx2class[classid_prediction]
    return {"prediction_class": class_predicted, "prediction_info": disease_info[classid_prediction]}


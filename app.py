from fastapi import FastAPI, File, UploadFile
import shutil
from datetime import datetime
import os
from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
from tensorflow.nn import sigmoid
from tensorflow.python.ops.numpy_ops import np_config
from json import dumps


app = FastAPI()
np_config.enable_numpy_behavior()
@app.post("/files")
# async def UploadImage(file: bytes = File(...)):
#     with open('image.jpg','wb') as image:
#         image.write(file)
#         image.close()
#     return 'got it'

async def image(image: UploadFile = File(...)):
#------------------Read and save image in uploads folder---------------------
    ts = datetime.timestamp(datetime.now())
    imgpath = os.path.join('uploads/', str(ts)+image.filename)
    outputpath = os.path.join('outputs/', os.path.basename(imgpath))
    with open(imgpath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

#---------------------Preprocess The Image------------------------------------
    img = load_img(imgpath, target_size = (224, 224))
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

#-----------------------Load Model--------------------------------------------
    my_loaded_model = tf.keras.models.load_model(
       ('my_model.h5'),custom_objects={'KerasLayer':hub.KerasLayer})

    #prediction = my_loaded_model.predict(imgpath)
#---------------------------- Predict using loaded Model------------------------
    pred = my_loaded_model.predict(img_array)
    score = sigmoid(pred[0])    
    model_score = round(max(score) * 100, 2)
    model_score = dumps(model_score.tolist())
    
    return {"Prediction : ":model_score}

# if >=50 : Has Pnemonia else No Pnemonia


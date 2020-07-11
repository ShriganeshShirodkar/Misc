from __future__ import division, print_function
from __future__ import division, print_function
import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, flash, request, redirect, url_for
import os
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from werkzeug import secure_filename
from PIL import Image
import numpy as np
from flask import send_from_directory
#------------------------------------------------------------------------

#%matplotlib inline

import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from tensorflow.keras.layers import Input
#from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#----------------------------------------------------------------------------

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import Sequential
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
#-------------------------------------------------------------------------------


UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['jpg','png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from flask_session import Session
#from flask import Flask, session
#sess = Session()


#app.config['SESSION_TYPE'] = 'memcached'
app.secret_key = "super secret key"

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Model saved with Keras model.save()
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)



MODEL_PATH = 'model-tgs-salt-1.h5'
import tensorflow as tf
model = load_model(MODEL_PATH,custom_objects={"mean_iou":mean_iou})
model._make_predict_function()          # Necessary

# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    X_t = np.zeros((1, 128, 128, 1), dtype=np.uint8)
    img = load_img(img_path)
    
    x = img_to_array(img)[:,:,1]
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    X_t[0] = x

    print('Done!')
    preds_test = model.predict(X_t, verbose=1)

# Threshold predictions

    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    return preds_test_t


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
#        if file and allowed_file(file.filename): --original
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #image = open_image(filename)
            image_url = url_for('uploaded_file', filename=filename)
            file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            global sess
            global graph
       
    
            with graph.as_default():
                set_session(sess)
                preds = model_predict(file_path, model)
            
            #preds = model_predict(file_path, model)
            print(preds)
            tmp = np.squeeze(preds).astype(np.float32)
            img = (np.dstack((tmp,tmp,tmp)))
            import cv2
            from random import seed
            from random import random
            # seed random number generator
            
            # generate random numbers between 0-1
            value = random()
    
            filename2= 'my'+str(value)+'.png'
            #W#filename5= 'my7585798.png'
            print(tmp)
            import matplotlib
            if np.max(preds)==1:
                prediction="Contains Salt"
            else:
                prediction="Does not contain salt"
            matplotlib.image.imsave(os.path.join(app.config['UPLOAD_FOLDER'], filename2), img)

            #cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)

            
            #img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            image_url2 = url_for('uploaded_filex', filename2=filename2)
            path=os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            return render_template("index4.html",image_url=image_url,x=image_url2,prediction=prediction)

    return render_template("index4.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/<filename2>')
def uploaded_filex(filename2):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename2)

# if __name__ == '__main__':
#     app.run(host='localhost',debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
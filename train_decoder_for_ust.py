#Implementing Universal Style Transfer as described in Li et al., arXiv:1705.08086v2
import os
import time
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import IPython.display

from PIL import Image

from tensorflow.python.keras.preprocessing import image as k_image
from tensorflow.python.keras import models

#Setup eager execution
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

#set image and content images, include path if not in same directory as this script
style_img = 'Madhubani.jpg'
content_img = 'AkankshaNaveen.jpg'

def load_img(img_path):
    max_size = 512
    img = Image.open(img_path)
    native_size = max(img.size)
    scale = max_size/native_size
    img = img.resize((round(img.size[0]*scale),round(img.size[1]*scale)),Image.ANTIALIAS)
    img = k_image.img_to_array(img)
    # tensorflow objects train when there is additional batch dimension, so add that 
    img = np.expand_dims(img,axis=0)
    
    return img

def imshow(img, title=None):
    #remove the batch dimension from img object
    img_disp = np.squeeze(img,axis=0)
    #normalize for display
    img_disp = img_disp.astype('uint8')
    plt.imshow(img_disp)
    if title is not None:
        plt.title(title)
    plt.imshow(img_disp)
    
plt.figure(figsize=(10,10))
content = load_img(content_img)
style = load_img(style_img)
plt.subplot(1,2,1)
imshow(content,'Content Image')
plt.subplot(1,2,2)    
imshow(style,'Style Image')
plt.show()    

def load_and_process_img(raw_img):
  img = load_img(raw_img)
  img_processed = tf.keras.applications.vgg19.preprocess_input(img)
  return img_processed

#to view images, we need to deprocess
def deprocess_img(img_processed):
    x = img_processed.copy()
    #squeeze batch dimension
    if len(x.shape) == 4:
        x = np.squeeze(x,axis=0)
    assert len(x.shape)==3
    #inverting preprocessing
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:, :, ::-1]
    
    x =np.clip(x,0,255).astype('uint8')
    return x

decoder_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_decoder_layers = len(decoder_layers)

def create_model_dcdr():
    #load Keras model for vgg19 pre-trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False)
    vgg.trainable = True
    #Get the outputs for required layers
    # Get output layers corresponding to style and content layers 
    model_outputs = [vgg.get_layer(name).output for name in decoder_layers]
    # Build model 
    return models.Model(vgg.input, model_outputs)

def get_features(model,init_image):
    image = load_and_process_img(init_image)
    
    
#defining loss as in eq.1 of the paper
def compute_loss(model_vgg,model_dcdr,init_img,vgg_weight):
    
    pixel_loss = tf.reduce_mean(tf.square(init_img - target_img))
    encdr_features = model_vgg(init_img);
    dcdr_features = model_dcdr(init_img);
    #feature loss
    for layer in vgg_init:
        feature_loss
    
    
    

    

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

import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as k_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

#Setup eager execution
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

#set image and content images, include path if not in same directory as this script
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
imshow(content,'Content Image')
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

decoder_layers = ['block5_conv1',
                'block4_conv1',
                'block3_conv1', 
                'block2_conv1', 
                'block1_conv1'
               ]

encoder_layers = ['block1_conv1',
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

def create_model_encdr():
    #load Keras model for vgg19 pre-trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False)
    vgg.trainable = False
    #Get the outputs for required layers
    # Get output layers corresponding to style and content layers 
    model_outputs = [vgg.get_layer(name).output for name in encoder_layers]
    # Build model 
    return models.Model(vgg.input, model_outputs)
    
#defining loss as in eq.1 of the paper
def compute_loss(model_encdr,model_dcdr,init_img,feature_weight):
    encdr_features = model_encdr(init_img)
    dcdr_features = model_dcdr(init_img)
    target_img = dcdr_features[-1]
    pixel_loss = tf.reduce_mean(tf.square(init_img - target_img))
    #feature loss
    feature_loss = 0
    for encdr_layer,dcdr_layer in zip(encdr_features,dcdr_features):
        feature_loss += tf.reduce_mean(tf.square(encdr_layer[0]-dcdr_layer[0]))
    
    return pixel_loss+feature_weight*feature_loss
    
def train_decoder(init_image,feature_weight=1,num_iterations = 1000):
    encdr = create_model_encdr()
    dcdr = create_model_dcdr()
    
    for layer in encdr.layers:
        layer.trainable = False
        
    for layer in dcdr.layers:
        layer.trainable = True
        
    init_img = load_and_process_img(init_image)
    init_img = tfe.Variable(init_img,dtype=tf.float32)

    #create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=0.1)
    
    #store the best result
    best_loss, best_img = float('inf'), None
    
    #For display
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start_time = time.time()
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    imgs = []
    
    for i in range(num_iterations):
        loss = compute_loss(encdr,dcdr,init_img,feature_weight)
        clipped = []
        #clip image values
        for ii in range(3):
            clipped.append(tf.clip_by_value(init_img[:,:,:,ii], min_vals[ii], max_vals[ii]))
            
        init_img.assign(tf.stack(clipped,axis=-1))
        
        if loss < best_loss:
            #Update best loss and best image
            best_loss = loss
            #.numpy() gives concrete array
            best_img = deprocess_img(init_img.numpy())
            
        if i % display_interval==0:
            start_time = time.time()
            plot_img = init_img.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait = True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('Total Loss: {:.4e},'
                  'Time: {:.4f}s'.format(loss,time.time()-start_time))
            
    print('Total Time: {:.4f}s'.format(time.time()-global_start_time))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        
    return best_img, best_loss




    

import h5py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.applications.xception import Xception
from keras.models import Model
    
def vgg(data,labels):
    input_shape = data.shape[1:]
    output_shape = labels.shape[1]
    vgg_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu',padding='same')(vgg_input)
    x = layers.Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu',padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x=layers.Flatten()(x)
    x=layers.Dropout(0.2)(x)
    x=layers.Dense(1000, activation='relu')(x)
    x=layers.Dense(100, activation='relu')(x)
    x=layers.Dense(output_shape, activation='linear')(x)
    
    model = models.Model(inputs=vgg_input, outputs=x)  
    model._name = 'vgg'
  
    return model
    
    

def vgg_small(data,labels):
    input_shape = data.shape[1:]
    output_shape = labels.shape[1]
    vgg_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu',padding='same')(vgg_input)
    x = layers.Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu',padding='same')(x)

    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu',padding='same')(x)
    x=layers.MaxPooling2D((2, 2))(x)
    
    x=layers.Flatten()(x)
    x=layers.Dropout(0.2)(x)
    x=layers.Dense(1000, activation='relu')(x)
    x=layers.Dense(500, activation='relu')(x)
    x=layers.Dense(300, activation='relu')(x)

    x=layers.Dense(100, activation='relu')(x)
    x=layers.Dense(50, activation='relu')(x)

    x=layers.Dense(output_shape, activation='linear')(x)
    
    model = models.Model(inputs=vgg_input, outputs=x)
    model._name = 'vgg_small'
   
    return model
    

def Xception_regression(data,labels):
    input_shape = data.shape[1:]
    output_shape = labels.shape[1]

    new_input = layers.Input(shape=(input_shape))
    
    upsample = layers.UpSampling3D(size=(1,1,3))(new_input)
    
    #Xception MUST have 3 channels input or crashes. 

    model = Xception(weights=None, include_top=False, input_shape=(input_shape[0],input_shape[1], 3),input_tensor=upsample)
    
    
    x = model.get_layer(index=len(model.layers)-2).output # cut off last layers, xception has 137000 parameters here
    
    #this is too big for us really. Lets max pool down to ~15,000

    x = layers.MaxPooling2D((3, 3))(x)

    #has 2048 channels, lets reduce to quarter this
    x = layers.Conv2D(512, (1, 1), activation='relu',padding='same')(x)

    #now output is approx 3000 elements

    x = layers.Flatten()(x) 
    x = layers.Dropout(0.2)(x)
    x=layers.Dense(1000)(x)

    x=layers.Dense(500)(x)
    x=layers.Dense(100)(x)

    x = layers.Dense(output_shape)(x)
    
    model = Model(inputs=model.input, outputs=x)

    model._name = 'Xception'
    return model
    
'''
This file will be ran on oberon. Output is saving the model.
'''

import sys
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath("__file__")))))

# sys.path.append(path)
sys.path.insert(0,path)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
matplotlib.use('agg')
from shack_hartmann.int_to_zern.Utils.models import Xception_regression, cnn, cnn_inceptionish, vgg, vgg_small, vgg_small_plus_larger_kernel
from shack_hartmann.int_to_zern.Utils.functions import pred_to_phasemap, DataGenerator, plot_training, normalize_data
import tensorflow as tf
import h5py
import shutil

class LSI_model:
    
    def __init__(self, hf, batch_size, model, lr = 0.0001):
        
        no_datapoints = len(hf['zernikes'])
        all_data = np.arange(no_datapoints)
        self.partition = {'train': all_data[:int(0.6*no_datapoints)],
                        'valid': all_data[int(0.6*no_datapoints):int(0.8*no_datapoints)],
                        'test': all_data[int(0.8*no_datapoints):]}
        
        

        self.model = model
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
            metrics=['mae'])
        
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        self.init_generators(hf=hf,batch_size = batch_size)


    def init_generators(self,hf,batch_size):
        self.training_generator = DataGenerator(self.partition['train'], hf, batch_size=batch_size, 
                        shuffle=False)
        self.validation_generator = DataGenerator(self.partition['valid'], hf, batch_size=batch_size, 
                        shuffle=False)
        self.test_generator = DataGenerator(self.partition['test'], hf, batch_size=batch_size, 
                        shuffle=False)


    def train_model(self,epochs = 10):
        history = self.model.fit(self.training_generator, epochs=epochs, 
                validation_data=self.validation_generator, callbacks = [self.callback])
        return history



if __name__ == '__main__':

    hf = h5py.File(path + "/shack_hartmann/zemax_sh_monster.h5", "r")
    
    model = vgg_small(data = np.array(hf['spots'][:2])[:,:,:,np.newaxis], labels = np.array(hf['zernikes'][:2]))
    
    batch_size = 16
    lr = 0.0001
    epochs = 10


    test = LSI_model(hf, batch_size=batch_size, model=model, lr = lr)
    
    new_dir = path+'/shack_hartmann/int_to_zern/model_data/'+model.name
    
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir) # will delete the old directory here beware...
    os.makedirs(new_dir)
    
    history = test.train_model(epochs=epochs)
    test.model.save(path + '/shack_hartmann/int_to_zern/model_data/'+ model.name +'.h5')
    
    plot_training(history,savefig=True, filename =new_dir +'/training_curve.png') 

    test_loss, test_acc = test.model.evaluate(test.test_generator, verbose=2)
    print((test_loss,test_acc))
    hf.close()

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
from shack_hartmann.int_to_zern.Utils.models import Xception_regression, vgg, vgg_small
from shack_hartmann.int_to_zern.Utils.functions import pred_to_phasemap, DataGenerator, plot_training, normalize_data,find_max_min
import tensorflow as tf
import h5py
import shutil

class LSI_model:
    
    def __init__(self, hf, batch_size, model, lr = 0.0001,maxmin = [0,0]):
        
        no_datapoints = len(hf['zernikes'])
        all_data = np.arange(no_datapoints)
        #self.partition = {'train': all_data[:int(0.6*no_datapoints)],
        #                'valid': all_data[int(0.6*no_datapoints):int(0.8*no_datapoints)],
        #                'test': all_data[int(0.8*no_datapoints):]}
        self.partition = {'train': all_data[:int(0.8*no_datapoints)],
                        'valid': all_data[int(0.8*no_datapoints):int(0.99*no_datapoints)],
                        'test': all_data[int(0.99*no_datapoints):]}
        

        self.model = model
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
            metrics=['mae'])
        
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        self.init_generators(hf=hf,batch_size = batch_size,maxmin =maxmin)


    def init_generators(self,hf,batch_size,maxmin =[0,0]):
        self.training_generator = DataGenerator(self.partition['train'], hf, batch_size=batch_size, 
                        shuffle=False,maxmin =maxmin)
        self.validation_generator = DataGenerator(self.partition['valid'], hf, batch_size=batch_size, 
                        shuffle=False,maxmin =maxmin)
        self.test_generator = DataGenerator(self.partition['test'], hf, batch_size=batch_size, 
                        shuffle=False,maxmin =maxmin)


    def train_model(self,epochs = 10):
        history = self.model.fit(self.training_generator, epochs=epochs, 
                validation_data=self.validation_generator, callbacks = [self.callback])
        return history



if __name__ == '__main__':
    
    print("Num GPUs:  ", len(tf.config.list_physical_devices('GPU')))
    exists = True
    hf = h5py.File(path + "/shack_hartmann/zemax_sh_monster.h5", "r")
    if not exists:
        model = vgg_small(data = np.array(hf['spots'][:2])[:,:,:,np.newaxis], labels = np.array(hf['zernikes'][:2]))
    else:
        model = tf.keras.models.load_model(path+'/shack_hartmann/int_to_zern/model_data/vgg_small.h5')

    batch_size = 64
    lr = 0.0001
    epochs = 5

    maxmin = find_max_min(hf) #first element is max, second is min


    test = LSI_model(hf, batch_size=batch_size, model=model, lr = lr, maxmin = maxmin)
    
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

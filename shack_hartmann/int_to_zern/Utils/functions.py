import os 
import sys
import h5py
import numpy as np
import LightPipes as lp
from LightPipes import *
import tensorflow.keras as keras
import matplotlib.pyplot as plt
# sys.path.append(path)

def find_max_min(h5):
    '''
    Need this so i can normalize properly. Basically loop through dataset and keep running totals of max and min.
    '''
    print('for large datasets this can take ~10mins')
     
    max_,min_ = 0,0
    
    #in sections of 1000, loop through.
    all = len(h5['spots'])
    
    for i in range(int(all/1000)):
        sub_max = np.max(h5['spots'][i*1000:(i+1)*1000])
        sub_min = np.min(h5['spots'][i*1000:(i+1)*1000])

        if sub_max > max_:
            max_ = sub_max
        if sub_min < min_:
            min_ = sub_min
    print((max_,min_))
    return [max_,min_]


def import_h5(filename,selection):
    h5f = h5py.File(sys.path[0] + '/ML/data/' + filename,'r')
            
    zernikes = np.array(h5f['all']['ground_truth']['zernikes'])
    phases = np.array(h5f['all']['ground_truth']['phases'])

    for i in selection[2:]:
        data = np.array(h5f['all']['ground_truth'][i])

    h5f.close()

    return zernikes, phases, data

def pred_to_phasemap(zernikes, N):
      λ = 800*lp.nm
      size=7*mm
      n_zernike = len(zernikes)
      F=lp.Begin(size,λ,N)
      zernike_poly = {i:zernikes[i] for i in range(1,n_zernike)} 

      for Noll in range(1, n_zernike):
          (nz, mz) = lp.noll_to_zern(Noll) #converting between int and zernike coeffs
          A = 1
          F = lp.Zernike(F, nz, mz, size, zernike_poly[Noll] * A, units='lam')

      F_ideal=CircAperture(F,size/2)
      return np.angle(F_ideal.field)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, hf_file, batch_size=32, 
                 shuffle=True,maxmin=[0,0]):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.hf = hf_file
        self.maxmin = maxmin

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        indexes = np.sort(indexes)
        act_indexes = self.list_IDs[indexes]

        X = np.array(self.hf['spots'][act_indexes])
        y = np.array(self.hf['zernikes'][act_indexes])
        
       # X[X>5] = 5 #to make more uniform
        X = normalize_data(X,self.maxmin) #(X-np.max(X,axis=(1,2))[:,None,None])/(np.max(X,axis=(1,2)) - np.min(X,axis=(1,2)))[:,None,None] #normalize data

        return X, y



def plot_training(history, log=False, savefig = False, filename =None):
    if log:
        plt.plot(np.log10(history.history['loss']), label='log_loss')
        plt.plot(np.log10(history.history['val_loss']), label = 'val_log_loss')
    else:
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    if savefig is True:
        if filename is None:
            raise ValueError('Specify filename')
        plt.savefig(filename,facecolor='w')

def normalize_data(X,maxmin):
    #return (X-np.min(X,axis=(1,2))[:,None,None])/(np.max(X,axis=(1,2)) - np.min(X,axis=(1,2)))[:,None,None] #normalize data
    return (X-maxmin[1])/(maxmin[0] - maxmin[1]) #normalize data
    
def psnr(true,pred,mask):
    '''
    input: true - N_examples x Dim_0 x Dim_1
           pred - N_examples x Dim_0 x Dim_1
           mask - Dim_0 x Dim_1
    '''

    sq_error = ((true * mask[None,:,:] - pred * mask[None,:,:])**2)

    mse_per_example = np.sum(sq_error,axis=(1,2))/(np.sum(mask==1)) 

    psnr_per_example = 10 * np.log10(np.max(true,axis=(1,2))**2 / mse_per_example)

    return np.mean(psnr_per_example)

def rmse(true,pred,mask):
    RMSE = np.sqrt(np.sum((true * mask[None,:,:] - pred * mask[None,:,:])**2)/(np.sum(mask==1) * true.shape[0]))
    return RMSE

import os 
import sys
import h5py
import numpy as np
import LightPipes as lp
from LightPipes import *
import tensorflow.keras as keras
import matplotlib.pyplot as plt
# sys.path.append(path)



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
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.hf = hf_file


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

        X = np.array(self.hf['LSI'][act_indexes])
        y = np.array(self.hf['zernikes'][act_indexes])
        
        X = normalize_data(X) #(X-np.max(X,axis=(1,2))[:,None,None])/(np.max(X,axis=(1,2)) - np.min(X,axis=(1,2)))[:,None,None] #normalize data

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

def normalize_data(X):
    return (X-np.min(X,axis=(1,2))[:,None,None])/(np.max(X,axis=(1,2)) - np.min(X,axis=(1,2)))[:,None,None] #normalize data


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

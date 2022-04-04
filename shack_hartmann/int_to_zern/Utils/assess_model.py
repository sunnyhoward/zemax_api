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
from shack_hartmann.int_to_zern.Utils.functions import pred_to_phasemap, DataGenerator, plot_training, normalize_data, rmse, psnr,find_max_min
import tensorflow as tf
import h5py

import LightPipes as lp
from LightPipes import *

from skimage.restoration import unwrap_phase

def test_set_analysis(test_data,test_labels,model,maxmin):
    '''here we calculate predictions on the test set.'''
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
            metrics=['mae'])
    
    #test_data[test_data>5] = 5
    test_data = normalize_data(test_data,maxmin)
     
    #test_data = test_data + np.max(np.abs(test_data))*0.2 * (np.random.rand(test_data.shape[0],test_data.shape[1],test_data.shape[2]) - 0.5)


    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    preds = model.predict(test_data)
    return preds


def zern_to_unwrapped(preds,test_labels,N):
    '''
    Here I convert from zernikes to the unwrapped phase maps.
    '''
    pred_phase = np.zeros((test_labels.shape[0],N,N))
    test_phase = np.zeros((test_labels.shape[0],N,N))

    print('converting zernike to wrapped phase map')
    for i in range(len(test_labels)):
        if (i% 100)==0:
            print(i)
        pred_phase[i] = pred_to_phasemap(preds[i],N)
        test_phase[i] = pred_to_phasemap(test_labels[i],N)

    unwrapped_pred_phase = np.zeros_like(pred_phase)
    unwrapped_test_phase = np.zeros_like(pred_phase)
    
    print('unwrapping')
    for i in range(len(test_labels)):
        if i%100 == 0:
            print(i)
        unwrapped_pred_phase[i] = unwrap_phase(pred_phase[i])
        unwrapped_test_phase[i] = unwrap_phase(test_phase[i])

    return pred_phase, test_phase, unwrapped_pred_phase, unwrapped_test_phase


def plot_wrapped(model,pred_phase,true_phase, mask, allele,path,n_plots):
    '''
    plot the wrapped phases and save in folder.
    '''
    RMSE = rmse(true_phase,pred_phase,mask)
    print('wrapped RMSE = '+str(RMSE))
    height=5
    length = int(height*n_plots/2)

    fig,ax = plt.subplots(2,n_plots,dpi=100,figsize=[length,height])
    for i in range(n_plots):
        ax[0,i].imshow(pred_phase[allele[i]]*mask,cmap='RdYlBu');ax[0, i].set_xticks([]);ax[0, i].set_yticks([])
        ax[1,i].imshow(mask*true_phase[allele[i]],cmap='RdYlBu');ax[1, i].set_xticks([]);ax[1, i].set_yticks([])

    ax[0, 0].set_ylabel('Reconstruction', fontsize=16)
    ax[1, 0].set_ylabel('Ground Truth', fontsize=16)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path+'/shack_hartmann/int_to_zern/model_data/'+model.name+'/wrapped_phases.png')

    
def plot_unwrapped(model,unwrapped_pred,unwrapped_true,mask,allele,path,n_plots):
   
    from scipy.optimize import leastsq
    def remove_offset(x, true_phase, recon_phase, mask):
        return np.mean(((recon_phase - true_phase)[mask==1]-x)**2)
   
    for i in range(len(unwrapped_pred)):
        offset=leastsq(remove_offset,0,(unwrapped_true[i], unwrapped_pred[i], mask))[0]
        unwrapped_pred[i] = unwrapped_pred[i] - offset
    #unwrapped_pred = unwrapped_pred - (np.min(unwrapped_pred,axis=(1,2)) + (np.max(unwrapped_pred,axis=(1,2)) - np.min(unwrapped_pred,axis=(1,2)))/2)[:,None,None]
    #unwrapped_true = unwrapped_true - (np.min(unwrapped_true,axis=(1,2)) + (np.max(unwrapped_true,axis=(1,2)) - np.min(unwrapped_true,axis = (1,2)))/2)[:,None,None]
    RMSE = rmse(unwrapped_true,unwrapped_pred,mask)
    print('unwrapped RMSE = ' + str(RMSE))
    
    height=5
    length = int(height*n_plots/2)

    fig,ax = plt.subplots(2,n_plots,dpi=100,figsize=[length,height])
    for i in range(n_plots):
        ax[0,i].imshow(mask*unwrapped_pred[allele[i]],cmap='RdYlBu');ax[0, i].set_xticks([]);ax[0, i].set_yticks([])
        ax[1,i].imshow(mask*unwrapped_true[allele[i]],cmap='RdYlBu');ax[1, i].set_xticks([]);ax[1, i].set_yticks([])

    ax[0, 0].set_ylabel('Reconstruction', fontsize=16)
    ax[1, 0].set_ylabel('Ground Truth', fontsize=16)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path+'/shack_hartmann/int_to_zern/model_data/'+model.name+'/unwrapped_phases.png')


def plot_zernikes(preds,test_labels,allele,path,n_plots):
    height=5
    length = int(height*n_plots/2)
    if len(preds[0]) == 36:
         fig,ax = plt.subplots(2,n_plots,dpi=100,figsize=[length,height])

         for i in range(n_plots):
             ax[0,i].imshow(preds[allele[i]].reshape((6,6)),cmap='RdYlBu');ax[0, i].set_xticks([]);ax[0, i].set_yticks([]);ax[0,i].set_title('example '+str(allele[i]))
             ax[1,i].imshow(test_labels[allele[i]].reshape((6,6)),cmap='RdYlBu');ax[1, i].set_xticks([]);ax[1, i].set_yticks([])

         ax[0, 0].set_ylabel('Reconstruction', fontsize=16)
         ax[1, 0].set_ylabel('Ground Truth', fontsize=16)
    
    elif len(preds[0]) == 45:
         fig,ax = plt.subplots(2,n_plots,dpi=100,figsize=[length,height])

         for i in range(n_plots):
             ax[0,i].imshow(preds[allele[i]].reshape((9,5)),cmap='RdYlBu');ax[0, i].set_xticks([]);ax[0, i].set_yticks([]);ax[0,i].set_title('example '+str(allele[i]))
             ax[1,i].imshow(test_labels[allele[i]].reshape((9,5)),cmap='RdYlBu');ax[1, i].set_xticks([]);ax[1, i].set_yticks([])

         ax[0, 0].set_ylabel('Reconstruction', fontsize=16)
         ax[1, 0].set_ylabel('Ground Truth', fontsize=16)
    
    elif len(preds[0]) == 15:
         fig,ax = plt.subplots(1,n_plots,dpi=100,figsize=[length,height])

         for i in range(n_plots):
             ax[i].plot(preds[allele[i]]);ax[i].set_xticks([]);ax[i].set_yticks([])
             ax[i].plot(test_labels[allele[i]]);ax[i].set_xticks([]);ax[i].set_yticks([])
             ax[i].legend(('preds','true'))
             

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path+'/shack_hartmann/int_to_zern/model_data/'+model.name+'/zernikes.png')



def create_mask(true_phase):
    #create mask
    border =0
    nx = len(true_phase[0])
    mask = np.zeros((nx,nx))
    for i in range(-nx//2,nx//2):
        for j in range(-nx//2,nx//2):
            if i**2 + j**2 < ((nx - 2*border)/2)**2:
                mask[i,j] = 1
    mask = np.fft.fftshift(mask)
    return mask
    


if __name__ == '__main__':
    
    hf = h5py.File(path+"/shack_hartmann/zemax_sh_monster.h5", "r")
    test_labels = np.array(hf['zernikes'][19900:])
    test_data=np.array(hf['spots'][19900:])
    maxmin = find_max_min(hf) #first element is max, second is min

    hf.close()

    model = tf.keras.models.load_model(path+'/shack_hartmann/int_to_zern/model_data/vgg_small.h5')
    new_dir = path+'/shack_hartmann/int_to_zern/model_data/'+model.name
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    nplots = 8

    predictions = test_set_analysis(test_data,test_labels,model,maxmin)
    N = test_data.shape[-1]
    
    pred_phase, test_phase, unwrapped_pred_phase, unwrapped_test_phase = zern_to_unwrapped(predictions,test_labels,N)
    mask= create_mask(test_phase)

    allele = np.arange(len(test_phase))
    np.random.shuffle(allele)

    plot_wrapped(model,pred_phase,test_phase, mask, allele,path,nplots)
    plot_unwrapped(model,unwrapped_pred_phase,unwrapped_test_phase,mask,allele,path,nplots)
    plot_zernikes(predictions,test_labels,allele,path,nplots)

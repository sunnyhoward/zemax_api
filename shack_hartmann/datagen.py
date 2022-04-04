import os 
import sys
import h5py
path = os.path.dirname(os.path.dirname(os.path.realpath("__file__")))
sys.path.insert(0,path)

import numpy as np
import matplotlib.pyplot as plt
import LightPipes as lp
from skimage.restoration import unwrap_phase
from tqdm import tqdm
from utils.api_functions import *
from utils.opt_functions import *
from shack_hartmann.sh_functions import *


#import connection
with open(r"\\alfs1.physics.ox.ac.uk\al\howards\Zemax\ZOS-API Projects\PythonZOSConnection\PythonZOSConnection.py") as f:
    code = compile(f.read(), r"\\alfs1.physics.ox.ac.uk\al\howards\Zemax\ZOS-API Projects\PythonZOSConnection\PythonZOSConnection.py", 'exec')
    exec(code)

fast_system(TheSystem) # this speeds up the program.

n_zernike = 15#45
n_lens = 67
w_lens = 0.15 #(mm)
aperture_value = n_lens*w_lens #this is what im saying is the width of the laser. we will cut out of it later.
crop_size = 47*w_lens #our detector width
Nx = 512
peak_size = 0 #0.2 #i dont really know what this means but it means the peaks have some width.


zernike_surf = TheSystem.LDE.GetSurfaceAt(1)
zernike_surf.SurfaceData.NumberOfTerms = n_zernike
zernike_surf.SurfaceData.NormRadius = aperture_value
# generate_zernike_file(zernike_coeffs,radius = 5) #here i create a file that can be read into zemax.

rotation_surf = TheSystem.LDE.GetSurfaceAt(5)
#
microlens_surf = TheSystem.LDE.GetSurfaceAt(4)
microlens_surf.SurfaceData.Par1.DoubleValue = n_lens
microlens_surf.SurfaceData.Par2.DoubleValue = n_lens #number of lens
microlens_surf.SurfaceData.Par3.DoubleValue = w_lens
microlens_surf.SurfaceData.Par4.DoubleValue = w_lens #lens width in mm

TheSystem.SystemData.Aperture.ApertureValue = aperture_value
TheSystem.SystemData.Aperture.ApodizationType = 1
TheSystem.SystemData.Aperture.ApodizationFactor = 1

wavelengths = np.arange(0.750,0.850,0.010)
rotations = np.linspace(-1,1,9)

weights = np.flip(np.arange(1,45,1))**5
p = weights/np.sum(weights) #create weights to prioritize low order abberations
Nx = 512
N=Nx
chunks = 200
chunksize = 100

for a in range(chunks):
    print(str(a/chunks * 100) + '%', end='\r')
    images = np.zeros([chunksize,N,N])
    zernikes = np.zeros([chunksize,n_zernike])

    for i in range(chunksize):
        


        # Generate abberations
        main_aberration = np.random.choice(np.arange(1,45), p = p)
        zernike_coeffs = np.random.rand(n_zernike)/(1+np.abs(np.arange(0,n_zernike,1)-main_aberration))**(2*np.random.rand(1))  
        rotation_amount = 0#np.random.choice(rotations) #add some rotation to detector

        set_zernikes(ZOSAPI, zernike_surf, n_zernike,zernike_coeffs)
        
        rotation_surf.SurfaceData.set_TiltAbout_Z(rotation_amount)
        zernike_coeffs = rot_zern_coeffs(old_coeffs = zernike_coeffs,rotation_deg = rotation_amount) #adjust zernikes accordingly.
        
        TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = np.random.choice(wavelengths) #set the wavelength


        images[i] = get_spots(ZOSAPI,TheSystem,Nx,rays=15,imagesize=crop_size,fieldsize = peak_size) # get spot pattern
        zernikes[i] = zernike_coeffs
        
    data = {'spots' : images,
            'zernikes' : zernikes}


    if a == 0:
        #initialize
        h5f = h5py.File(('zemax_match_ATLAS_norotationorpeaksize.h5'), 'w')
            

        h5f.create_dataset('zernikes', data=data['zernikes'], compression="gzip", chunks=True, maxshape=(None,n_zernike))
        h5f.create_dataset('spots', data=data['spots'], compression="gzip", chunks=True, maxshape=(None,N,N))
    else:
        
        h5f['zernikes'].resize((h5f['zernikes'].shape[0] + data['zernikes'].shape[0]),axis=0)
        h5f['zernikes'][-data['zernikes'].shape[0]:] = data['zernikes']
        
        h5f['spots'].resize((h5f['spots'].shape[0] + data['spots'].shape[0]),axis=0)
        h5f['spots'][-data['spots'].shape[0]:] = data['spots']



h5f.close()
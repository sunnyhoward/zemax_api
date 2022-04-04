import sys
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath("__file__")))))
import h5py
# sys.path.append(path)
sys.path.insert(0,path)
import numpy as np
from shack_hartmann.int_to_zern.Utils.models import Xception_regression, vgg, vgg_small


hf = h5py.File(path + "/shack_hartmann/zemax_sh_monster.h5", "r")
model = Xception_regression(data = np.array(hf['spots'][:2])[:,:,:,np.newaxis], labels = np.array(hf['zernikes'][:2]))

print(model.summary())

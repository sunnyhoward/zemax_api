import numpy as np
import matplotlib.pyplot as plt
import LightPipes as lp


def current_zernikes(ZOSAPI, zernike_surf, n_zernikes):
    '''
    Function to see the current zernikes in the system.
    zernike_surf - the surface with the zernike addition
    n_zernikes - number of zernikes.
    '''

    annoyingstring = 'zernike_surf.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par'
    zero = 15
    for i in range(n_zernikes):

        bla = eval(annoyingstring+str(i + zero)+').DoubleValue') 
        print(bla)

    return


def set_zernikes(ZOSAPI, zernike_surf, n_zernikes, zernike_coeffs):
    '''
    see above. Really i want to do this by creating a file and uploading as i imagine its way faster.
    '''


    annoyingstring = 'zernike_surf.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par'

    zero = 15
    for i in range(n_zernikes):

        bla = eval(annoyingstring+str(i + zero)+')') 
        bla.set_DoubleValue(zernike_coeffs[i])

def get_spots(ZOSAPI,TheSystem,Nx):
    '''
    generate spot pattern
    '''

    gia = TheSystem.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.GeometricImageAnalysis)

    gia.GetSettings().set_ShowAs(1)
    gia.GetSettings().set_NumberOfPixels(Nx)
    gia.GetSettings().set_ImageSize(7)
    gia.GetSettings().set_TotalWatts(10)

    gia.ApplyAndWaitForCompletion()

    gia_results = gia.GetResults()

    gia.ApplyAndWaitForCompletion()

    spots = np.array(list(gia_results.GetDataGrid(0).Values)).reshape(-1,Nx)
    gia.Close()

    return spots


def get_wavefront(ZOSAPI,TheSystem,Nx):
    '''
    generate wavefront pattern
    '''
    MyWavefrontMap = TheSystem.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.WavefrontMap)

    MyWavefrontMapSettings = MyWavefrontMap.GetSettings()

    # MyWavefrontMapSettings.Surface = 
    MyWavefrontMapSettings.set_Sampling(6)
    MyWavefrontMap.ApplyAndWaitForCompletion()
    MyWavefrontMapResults = MyWavefrontMap.GetResults()

    MyWavefrontMapGrid = MyWavefrontMapResults.GetDataGrid(0).Values
    wavefront = np.array(list(MyWavefrontMapGrid)).reshape(-1,Nx)

    MyWavefrontMap.Close()
    
    return wavefront





def generate_zernike_file(zernike_coeffs,radius):
    '''
    see below for file format.
    https://support.zemax.com/hc/en-us/articles/1500005575422-How-to-model-a-black-box-optical-system-using-Zernike-coefficients
    '''
    f = open(r"\\alfs1.physics.ox.ac.uk\al\howards\Zemax\Objects\Grid Files\zernikes.dat",'w')

    f.write(str(len(zernike_coeffs)) + "\n")
    f.write(str(radius) + "\n")

    for i in range(len(zernike_coeffs)):
        f.write(str(zernike_coeffs[i]) + "\n")
    f.close()



def rot_zern_coeffs(old_coeffs,rotation_deg):
    '''
    Function to rotate zernike coefficients by a certain angle. Useful for generating training data.
    See 'Description of Zernike Polynomials' - Jim Schwiegerling
    '''
    rotation = rotation_deg * np.pi/180
    new_coeffs = np.zeros_like(old_coeffs)
    n_zernike = len(old_coeffs)

    #below i check that the n_zernikes selected means a fullset of nz
    noll_set = np.zeros(20)
    for i in range(20):
        noll_set[i] =  noll_set[i-1] + i+1

    if n_zernike not in noll_set:
        raise ValueError('You must pick a value of n_zernike that allows for a full set of zernike coefficients. ie 1,3,6,10,15...')

    Noll = 1
    while Noll<n_zernike+1:
        (nz, mz) = lp.noll_to_zern(Noll) #orders them in pairs
        
        if mz == 0:
            new_coeffs[Noll-1] = old_coeffs[Noll-1] #all ones with m=0 are radially symmetric anyway.
            
        elif mz>0: #if mz>0 then the mz for the next Noll is -|mz|
                            
            new_coeffs[Noll-1] = old_coeffs[Noll-1] * np.cos(np.abs(mz) * rotation) - old_coeffs[Noll] * np.sin(np.abs(mz) * rotation)
            new_coeffs[Noll] = old_coeffs[Noll-1] * np.sin(np.abs(mz) * rotation) + old_coeffs[Noll] * np.cos(np.abs(mz) * rotation)

            Noll+=1 #skip an extra one
        
        elif mz<0: #if mz<0 then the mz for the next Noll is |mz|
                            
            new_coeffs[Noll-1] = old_coeffs[Noll] * np.sin(np.abs(mz) * rotation) + old_coeffs[Noll-1] * np.cos(np.abs(mz) * rotation)
            new_coeffs[Noll] = old_coeffs[Noll] * np.cos(np.abs(mz) * rotation) - old_coeffs[Noll-1] * np.sin(np.abs(mz) * rotation)

            Noll+=1 #skip an extra one
           
        Noll += 1

    
    return new_coeffs
import numpy as np
import matplotlib.pyplot as plt

def double2array(data):
    return np.array(list(data))

def analyse_MTF(System,ZOSAPI,max_freq):
    mtf = System.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.FftMtf);
    mtf.Settings.MaximumFrequency = max_freq

    mtf.ApplyAndWaitForCompletion()

    mtf_res = mtf.GetResults()
    length = mtf_res.DataSeries.Length

    angles = [0,3.5,5]
    types = ['tan','sag']
    leg = []
    plt.figure(dpi=200)
    for i in range(length):
        xdat = double2array(mtf_res.DataSeries.GetValue(i).XData.Data)
        ydat = double2array(mtf_res.DataSeries.GetValue(i).YData.Data)

        # y consists of half data tagential and half sagittal. 
        ytan = ydat[::2]
        ysag = ydat[1::2]



        plt.plot(xdat,ytan,xdat,ysag)
        leg.append(str(angles[i]) + types[0])
        leg.append(str(angles[i]) + types[1])
    
    plt.legend((leg))
    plt.xlabel('Spatial Freq');
    plt.ylabel('Modulation');

    plt.xlim(0,max_freq)
    plt.ylim(-0.05,1.05)
    
    mtf_res.Disconnect()
    mtf.Close()
import numpy as np
from scipy.signal import hilbert
from scipy.signal import savgol_filter

def upsampled_heart_signal(fnirs_od_signal):
# This function take as inputs the preprocessed optical density (od) signals from fNIRS in the shape (n_channels, n_samples).
# These od signals have to be bandpass filtered in the range of the heart beat per second (bps) of interest.
# For example, for 60 to 120 bps the cutoof frequencies are 1 Hz and 2 Hz.
# With all the od signals from the fNIRS channels we can compute the heart signal with the original sampling frequency of the device.
# First, we normalized the od signals with their envelopes.
# Then, we use all the samples from the fNIRS channels considering the light sequence of the sources.
# This is done by interleaving the channel samples as they correspond to each source in order.
# In each time point we are going to have several samples, thus we compute the mean of the samples at each time point.
# Finally, the curve is smoothed with a Savitzky-Golay filter.


    Fs_device = 62.5 #Hz ->16ms on each source
    Fs_channel = 7.8125 #Hz 
    n_sources = 8 #There are 8 sources sequentilay switched on
    n_samples_ch = fnirs_od_signal.shape[1]
    n_samples_total = n_samples_ch*n_sources 


    signal_env=np.abs(hilbert(fnirs_od_signal,axis=1)) #Envelpe with Hilbert Transform
    fnirs_od_signal_norm = fnirs_od_signal / signal_env #Normalized od

    #fNIRS sources sequence correspondence of sampling time points 
    time1 = np.arange(0,n_samples_ch/Fs_channel,1/Fs_channel)
    time2 = np.arange(1/Fs_device ,n_samples_ch/Fs_channel+1/Fs_device ,1/Fs_channel)
    time3 = np.arange(2/Fs_device ,n_samples_ch/Fs_channel+2/Fs_device ,1/Fs_channel)
    time4 = np.arange(3/Fs_device ,n_samples_ch/Fs_channel+3/Fs_device ,1/Fs_channel)
    time5 = np.arange(4/Fs_device ,n_samples_ch/Fs_channel+4/Fs_device ,1/Fs_channel)
    time6 = np.arange(5/Fs_device ,n_samples_ch/Fs_channel+5/Fs_device ,1/Fs_channel)
    time7 = np.arange(6/Fs_device ,n_samples_ch/Fs_channel+6/Fs_device ,1/Fs_channel)
    time8 = np.arange(7/Fs_device ,n_samples_ch/Fs_channel+7/Fs_device ,1/Fs_channel)
    pos_index_1 = time1*Fs_device 
    pos_index_2 = time2*Fs_device 
    pos_index_3 = time3*Fs_device 
    pos_index_4 = time4*Fs_device 
    pos_index_5 = time5*Fs_device 
    pos_index_6 = time6*Fs_device 
    pos_index_7 = time7*Fs_device 
    pos_index_8 = time8*Fs_device 

    #fNIRS source-channels correspondence 
    ch_s1_hbo = [1,3]
    ch_s2_hbo = [5,7,9]
    ch_s3_hbo = [11,13,15]
    ch_s4_hbo = [17,19]
    ch_s5_hbo = [21,23]
    ch_s6_hbo = [25,27,29]
    ch_s7_hbo = [31,33,35]
    ch_s8_hbo = [37,39]
    ch_s1_hbr = [0,2]
    ch_s2_hbr = [4,6,8]
    ch_s3_hbr = [10,12,14]
    ch_s4_hbr = [16,18]
    ch_s5_hbr = [20,22]
    ch_s6_hbr = [24,26,28]
    ch_s7_hbr = [30,32,34]
    ch_s8_hbr = [36,38]


        
    heart_signal_upsampled = np.zeros((n_samples_total)) 

    heart_signal_upsampled[np.round(pos_index_1).astype(int)]=np.mean(fnirs_od_signal_norm[ch_s1_hbo+ch_s1_hbr],axis=0)  
    heart_signal_upsampled[np.round(pos_index_2).astype(int)]=np.mean(fnirs_od_signal_norm[ch_s2_hbo+ch_s2_hbr],axis=0)  
    heart_signal_upsampled[np.round(pos_index_3).astype(int)]=np.mean(fnirs_od_signal_norm[ch_s3_hbo+ch_s3_hbr],axis=0) 
    heart_signal_upsampled[np.round(pos_index_4).astype(int)]=np.mean(fnirs_od_signal_norm[ch_s4_hbo+ch_s4_hbr],axis=0) 
    heart_signal_upsampled[np.round(pos_index_5).astype(int)]=np.mean(fnirs_od_signal_norm[ch_s5_hbo+ch_s5_hbr],axis=0)  
    heart_signal_upsampled[np.round(pos_index_6).astype(int)]=np.mean(fnirs_od_signal_norm[ch_s6_hbo+ch_s6_hbr],axis=0)  
    heart_signal_upsampled[np.round(pos_index_7).astype(int)]=np.mean(fnirs_od_signal_norm[ch_s7_hbo+ch_s7_hbr],axis=0)
    heart_signal_upsampled[np.round(pos_index_8).astype(int)]=np.mean(fnirs_od_signal_norm[ch_s8_hbo+ch_s8_hbr],axis=0)  

    #Savitzky-Golay filter 
    heart_signal_upsampled_smooth =savgol_filter(heart_signal_upsampled, 15, 2)
    
    return heart_signal_upsampled,heart_signal_upsampled_smooth
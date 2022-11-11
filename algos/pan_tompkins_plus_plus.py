# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:37:40 2022

@author: Niaz
"""
import numpy as np
from scipy.interpolate import interp1d
import peakutils
from scipy import signal
import six
import scipy.signal as sig



class Pan_Tompkins_Plus_Plus():

    """
           This is the Code for the Pan-Tompkins++ algorithm that is an improved Pan-Tompkins algorithm (https://github.com/Fabrizio1994/ECGClassification/blob/master/rpeakdetection/pan_tompkins/pan.py)
            
            
            Inputs
            ----------
             ecg : raw ecg vector signal 1d signal
             fs : sampling frequency 

            Outputs
            -------
            qrs_amp_raw : amplitude of R waves amplitudes
            qrs_i_raw : index of R waves
            delay : number of samples which the signal is delayed due to the filtering

        """
    def get_name(self):
        return "pan_tompkins_plus_plus"
    
    
    def rpeak_detection(self, ecg, fs):

        ''' Initialize '''

        delay = 0
        skip = 0                    # Becomes one when a T wave is detected
        m_selected_RR = 0
        mean_RR = 0
        ser_back = 0


        ''' Noise Cancelation (Filtering) (5-18 Hz) '''

        if fs == 200:
            ''' Remove the mean of Signal '''
			#If fs=200 keep frwquency 5-12 Hz otherwise 5-18 Hz
            ecg = ecg - np.mean(ecg)  

            Wn = 12*2/fs
            N = 3
            a, b = signal.butter(N, Wn, btype='lowpass')
            ecg_l = signal.filtfilt(a, b, ecg)
            
            ecg_l = ecg_l/np.max(np.abs(ecg_l)) #Normalize by dividing high value. That reduce time of calculation

            Wn = 5*2/fs
            N = 3                                           # Order of 3 less processing
            a, b = signal.butter(N, Wn, btype='highpass')             # Bandpass filtering
            ecg_h = signal.filtfilt(a, b, ecg_l, padlen=3*(max(len(a), len(b))-1))
            ecg_h = ecg_h/np.max(np.abs(ecg_h))  #Normalize by dividing high value. That reduce time of calculation

        else:
            ''' Band Pass Filter for noise cancelation of other sampling frequencies (Filtering)'''
            f1 = 5 #3 #5                                          # cutoff low frequency to get rid of baseline wander
            f2 = 18 #25  #15                                         # cutoff frequency to discard high frequency noise
            Wn = [f1*2/fs, f2*2/fs]                         # cutoff based on fs
            N = 3                                           
            a, b = signal.butter(N=N, Wn=Wn, btype='bandpass')   # Bandpass filtering
            ecg_h = signal.filtfilt(a, b, ecg, padlen=3*(max(len(a), len(b)) - 1))
                       
            ecg_h = ecg_h/np.max(np.abs(ecg_h))

        vector = [1, 2, 0, -2, -1]
        if fs != 200:
            int_c = 160/fs
            b = interp1d(range(1, 6), [i*fs/8 for i in vector])(np.arange(1, 5.1, int_c))  
																						
        else:
            b = [i*fs/8 for i in vector]      

        ecg_d = signal.filtfilt(b, 1, ecg_h, padlen=3*(max(len(a), len(b)) - 1))

        ecg_d = ecg_d/np.max(ecg_d)


        ''' Squaring nonlinearly enhance the dominant peaks '''

        ecg_s = ecg_d**2
        
        #Smooting
        sm_size = int(0.06 * fs)       
        ecg_s = smoother(signal=ecg_s, kernel='flattop', size=sm_size, mirror=True)


        temp_vector = np.ones((1, round(0.150*fs)))/round(0.150*fs) # 150ms moving window, widest possible QRS width
        temp_vector = temp_vector.flatten()
        ecg_m = np.convolve(ecg_s, temp_vector)  #Convolution signal and moving window sample

        delay = delay + round(0.150*fs)/2


        pks = []
        locs = peakutils.indexes(y=ecg_m, thres=0, min_dist=round(0.231*fs))  #Find all the peaks apart from previous peak 231ms, peak indices
        for val in locs:
            pks.append(ecg_m[val])     #Peak magnitudes
 
        ''' Initialize Some Other Parameters '''
        LLp = len(pks)

        ''' Stores QRS with respect to Signal and Filtered Signal '''
        qrs_c = np.zeros(LLp)           # Amplitude of R peak in convoluted (after moving window) signal
        qrs_i = np.zeros(LLp)           # Index of R peak in convoluted (after moving window) signal
        qrs_i_raw = np.zeros(LLp)       # Index of R peak in filtered (before derivative and moving windoe) signal 
        qrs_amp_raw = np.zeros(LLp)     # Amplitude of R in filtered signal
        ''' Noise Buffers '''
        nois_c = np.zeros(LLp)
        nois_i = np.zeros(LLp)

        ''' Buffers for signal and noise '''

        SIGL_buf = np.zeros(LLp)
        NOISL_buf = np.zeros(LLp)
        THRS_buf = np.zeros(LLp)
        SIGL_buf1 = np.zeros(LLp)
        NOISL_buf1 = np.zeros(LLp)
        THRS_buf1 = np.zeros(LLp)

        ''' Initialize the training phase (2 seconds of the signal) to determine the THR_SIG and THR_NOISE '''
		#Threshold of signal after moving average operation; Take first 2s window max peak to set initial Threshold
        THR_SIG = np.max(ecg_m[:2*fs+1])*1/3                 # Threshold-1 (paper) #0.33 of the max amplitude 
        THR_NOISE = np.mean(ecg_m[:2*fs+1])*1/2              #Threshold-2 (paper) # 0.5 of the mean signal is considered to be noise
        SIG_LEV = THR_SIG                         #SPK for convoluted (after moving window) signal
        NOISE_LEV = THR_NOISE                     #NPK for convoluted (after moving window) signal


        ''' Initialize bandpath filter threshold (2 seconds of the bandpass signal) '''
		#Threshold of signal before derivative and moving average operation, just after 5-18 Hz filtering
        THR_SIG1 = np.max(ecg_h[:2*fs+1])*1/3               #Threshold-1
        THR_NOISE1 = np.mean(ecg_h[:2*fs+1])*1/2            #Threshold-2
        SIG_LEV1 = THR_SIG1                                 # Signal level in Bandpassed filter; SPK for filtered signal
        NOISE_LEV1 = THR_NOISE1                             # Noise level in Bandpassed filter; NPK for filtered signal



        ''' Thresholding and decision rule '''

        Beat_C = 0       #Beat count for convoluted signal
        Beat_C1 = 0      #Beat count for filtred signal
        Noise_Count = 0
        Check_Flag=0
        for i in range(LLp):
            ''' Locate the corresponding peak in the filtered signal '''

            if locs[i] - round(0.150*fs) >= 1 and locs[i] <= len(ecg_h): 
                temp_vec = ecg_h[locs[i] - round(0.150*fs):locs[i]+1]     # Find the values from the preceding 150ms of the peak
                y_i = np.max(temp_vec)                      #Find the max magnitude in that 150ms window
                x_i = list(temp_vec).index(y_i)             #Find the index of the max value with respect to (peak-150ms) starts as 0 index
            else:
                if i == 0:
                    temp_vec = ecg_h[:locs[i]+1]
                    y_i = np.max(temp_vec)
                    x_i = list(temp_vec).index(y_i)
                    ser_back = 1
                elif locs[i] >= len(ecg_h):
                    temp_vec = ecg_h[int(locs[i] - round(0.150*fs)):] #c
                    y_i = np.max(temp_vec)
                    x_i = list(temp_vec).index(y_i)


            ''' Update the Hearth Rate '''
            if Beat_C >= 9:
                diffRR = np.diff(qrs_i[Beat_C-9:Beat_C])            # Calculate RR interval of recent 8 heart beats (taken from R peaks)
                mean_RR = np.mean(diffRR)                           # Calculate the mean of 8 previous R waves interval
                comp = qrs_i[Beat_C-1] - qrs_i[Beat_C-2]              # Latest RR
                
                m_selected_RR = mean_RR                         #The latest regular beats mean
			
            ''' Calculate the mean last 8 R waves '''
            if bool(m_selected_RR):
                test_m = m_selected_RR                              #if the regular RR available use it
            elif bool(mean_RR) and m_selected_RR == 0:
                test_m = mean_RR
            else:
                test_m = 0

            #If no R peaks in 1.4s then check with the reduced Threshold    
            if (locs[i] - qrs_i[Beat_C-1]) >= round(1.4*fs):     
                    
                  temp_vec = ecg_m[int(qrs_i[Beat_C-1] + round(0.360*fs)):int(locs[i])+1] #Search after 360ms of previous QRS to current peak
                  if temp_vec.size:
                      pks_temp = np.max(temp_vec) #search back and locate the max in the interval
                      locs_temp = list(temp_vec).index(pks_temp)
                      locs_temp = qrs_i[Beat_C-1] + round(0.360*fs) + locs_temp
                      

                      if pks_temp > THR_NOISE*0.2:  #Check with 20% of the noise threshold
  												
                          Beat_C = Beat_C + 1
                          if (Beat_C-1)>=LLp:
                              break
                          qrs_c[Beat_C-1] = pks_temp   
                          qrs_i[Beat_C-1] = locs_temp
  
  
                          ''' Locate in Filtered Signal '''
  						#Once we find the peak in convoluted signal, we will search in the filtered signal for max peak with a 150 ms window before that location
                          if locs_temp <= len(ecg_h):
                             
                              temp_vec = ecg_h[int(locs_temp-round(0.150*fs))+1:int(locs_temp)+2]  
                              y_i_t = np.max(temp_vec)
                              x_i_t = list(temp_vec).index(y_i_t)
                          else:
                              temp_vec = ecg_h[int(locs_temp-round(0.150*fs)):]
                              y_i_t = np.max(temp_vec)
                              x_i_t = list(temp_vec).index(y_i_t)
                

                          if y_i_t > THR_NOISE1*0.2:
                              Beat_C1 = Beat_C1 + 1
                              if (Beat_C1-1)>=LLp:
                                  break
                              temp_value = locs_temp - round(0.150*fs) + x_i_t
                              qrs_i_raw[Beat_C1-1] = temp_value                           
                              qrs_amp_raw[Beat_C1-1] = y_i_t                                 
                              
                              SIG_LEV1 = 0.75 * y_i_t + 0.25 *SIG_LEV1                     
  																						
  
                          not_nois = 1
                          
                          SIG_LEV = 0.75 * pks_temp + 0.25 *SIG_LEV           
                  else:
                      not_nois = 0
            
            
            elif bool(test_m):  #Check for missed QRS if no QRS is detected in 166 percent of
								#the current average RR interval or 1s after the last detected QRS. the maximal peak detected in
								#that time interval that lies Threshold1 and Threshold-3 (paper) is considered to be a possible QRS complex

                if ((locs[i] - qrs_i[Beat_C-1]) >= round(1.66*test_m)) or ((locs[i] - qrs_i[Beat_C-1]) > round(1*fs)):     #it shows a QRS is missed
                                        
                    temp_vec = ecg_m[int(qrs_i[Beat_C-1] + round(0.360*fs)):int(locs[i])+1] #Search after 360ms of previous QRS to current peak
                    if temp_vec.size:
                        pks_temp = np.max(temp_vec) #search back and locate the max in the interval
                        locs_temp = list(temp_vec).index(pks_temp)
                        locs_temp = qrs_i[Beat_C-1] + round(0.360*fs) + locs_temp
                        
                        #Consider signal between the preceding 3 QRS complexes and the following 3 peaks to calculate Threshold-3 (paper)
                        
                        THR_NOISE_TMP=THR_NOISE
                        if i<(len(locs)-3):
                            temp_vec_tmp=ecg_m[int(qrs_i[Beat_C-3] + round(0.360*fs)):int(locs[i+3])+1] #values between the preceding 3 QRS complexes and the following 3 peaks
                            THR_NOISE_TMP =0.5*THR_NOISE+0.5*( np.mean(temp_vec_tmp)*1/2) #Calculate Threshold3 
                        
                        if pks_temp > THR_NOISE_TMP:  #If max peak in that range greater than Threshold3 mark that as a heart beat
    												
                            Beat_C = Beat_C + 1
                            if (Beat_C-1)>=LLp:
                                break
                            qrs_c[Beat_C-1] = pks_temp   #Mark R peak in the convoluted signal
                            qrs_i[Beat_C-1] = locs_temp
    
    
                            ''' Locate in Filtered Signal '''
    						#Once we find the peak in convoluted signal, we will search in the filtered signal for max peak with a 150 ms window before that location
                            if locs_temp <= len(ecg_h):
                            
                                temp_vec = ecg_h[int(locs_temp-round(0.150*fs))+1:int(locs_temp)+2]  
                                y_i_t = np.max(temp_vec)
                                x_i_t = list(temp_vec).index(y_i_t)
                            else:
                                temp_vec = ecg_h[int(locs_temp-round(0.150*fs)):]
                                y_i_t = np.max(temp_vec)
                                x_i_t = list(temp_vec).index(y_i_t)
                            
                    
                            ''' Band Pass Signal Threshold '''
                            THR_NOISE1_TMP=THR_NOISE1
                            if i<(len(locs)-3):
                                temp_vec_tmp=ecg_h[int(qrs_i[Beat_C-3] + round(0.360*fs)-round(0.150*fs)+1):int(locs[i+3])+1]
                                THR_NOISE1_TMP =0.5*THR_NOISE1+0.5*( np.mean(temp_vec_tmp)*1/2)
                            if y_i_t > THR_NOISE1_TMP:
                                Beat_C1 = Beat_C1 + 1
                                if (Beat_C1-1)>=LLp:
                                    break
                                temp_value = locs_temp - round(0.150*fs) + x_i_t
                                qrs_i_raw[Beat_C1-1] = temp_value                           # R peak marked with index in filtered signal
                                qrs_amp_raw[Beat_C1-1] = y_i_t                                 # Amplitude of that R peak
                                
                                SIG_LEV1 = 0.75 * y_i_t + 0.25 *SIG_LEV1                     
    																						
    
                            not_nois = 1
                             #Changed- For missed R peaks- Update THR
                            SIG_LEV = 0.75 * pks_temp + 0.25 *SIG_LEV          
                    else:
                        not_nois = 0
                else:
                    not_nois = 0
                    
                    

            ''' Find noise and QRS Peaks '''

            if pks[i] >= THR_SIG:
                ''' if NO QRS in 360 ms of the previous QRS or in 50 percent of
								the current average RR interval, See if T wave '''
              
                if Beat_C >= 3:
                    if bool(test_m):
                        if (locs[i] - qrs_i[Beat_C-1]) <= round(0.5*test_m): #Check 50 percent of the current average RR interval
								
                            Check_Flag=1
                    if (locs[i] - qrs_i[Beat_C-1] <= round(0.36*fs)) or Check_Flag==1:  
                       
                        temp_vec = ecg_m[locs[i]-round(0.07*fs):locs[i]+1]
                        Slope1 = np.mean(np.diff(temp_vec))          # mean slope of the waveform at that position
                        temp_vec = ecg_m[int(qrs_i[Beat_C-1] - round(0.07*fs)) - 1 : int(qrs_i[Beat_C-1])+1]
                        Slope2 = np.mean(np.diff(temp_vec))        # mean slope of previous R wave

                        if np.abs(Slope1) <= np.abs(0.6*Slope2):          # slope less then 0.6 of previous R; checking if it is noise
                            Noise_Count = Noise_Count + 1
                            nois_c[Noise_Count] = pks[i]
                            nois_i[Noise_Count] = locs[i]
                            skip = 1                                              # T wave identification
                        else:
                            skip = 0

                ''' Skip is 1 when a T wave is detected '''
                if skip == 0:
                    Beat_C = Beat_C + 1
                    if (Beat_C-1)>=LLp:
                        break
                    qrs_c[Beat_C-1] = pks[i]     #Mark as R peak in the convoluted signal
                    qrs_i[Beat_C-1] = locs[i]


                    ''' Band pass Filter check threshold '''

                    if y_i >= THR_SIG1:
                        Beat_C1 = Beat_C1 + 1            #Mark as R peak in the filtered signal
                        if (Beat_C1-1)>=LLp:
                            break
                        if bool(ser_back):
                            # +1 to agree with Matlab implementation
                            temp_value = x_i + 1
                            qrs_i_raw[Beat_C1-1] = temp_value
                        else:
                            temp_value = locs[i] - round(0.150*fs) + x_i
                            qrs_i_raw[Beat_C1-1] = temp_value

                        qrs_amp_raw[Beat_C1-1] = y_i

                        SIG_LEV1 = 0.125*y_i + 0.875*SIG_LEV1


                    SIG_LEV = 0.125*pks[i] + 0.875*SIG_LEV


            elif THR_NOISE <= pks[i] and pks[i] < THR_SIG:
                NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                NOISE_LEV = 0.125*pks[i] + 0.875 * NOISE_LEV

            elif pks[i] < THR_NOISE:           #If less than noise threshold (Threshold-2) mark as noise
                nois_c[Noise_Count] = pks[i]
                nois_i[Noise_Count] = locs[i]
                Noise_Count = Noise_Count + 1


                NOISE_LEV1 = 0.125*y_i +0.875 *NOISE_LEV1
                NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV

            ''' Adjust the threshold with SNR '''

            if NOISE_LEV != 0 or SIG_LEV != 0:
                THR_SIG = NOISE_LEV + 0.25 * (np.abs(SIG_LEV - NOISE_LEV))  #Calculate Threshold-1 for convoluted signal; above this R peak
                THR_NOISE = 0.4* THR_SIG                                   #Calculate Threshold-2 for convoluted signal; below this Noise
			
            ''' Adjust the threshold with SNR for bandpassed signal '''

            if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
                THR_SIG1 = NOISE_LEV1 + 0.25*(np.abs(SIG_LEV1 - NOISE_LEV1)) #Calculate Threshold-1  for filtered signal; above this R peak
                THR_NOISE1 = 0.4* THR_SIG1                   #Calculate Threshold-2 for filtered signal; below this Noise


            ''' take a track of thresholds of smoothed signal '''

            SIGL_buf[i] = SIG_LEV
            NOISL_buf[i] = NOISE_LEV
            THRS_buf[i] = THR_SIG

            ''' take a track of thresholds of filtered signal '''

            SIGL_buf1[i] = SIG_LEV1
            NOISL_buf1[i] = NOISE_LEV1
            THRS_buf1[i] = THR_SIG1

            ''' reset parameters '''

            skip = 0
            not_nois = 0
            ser_back = 0
            Check_Flag=0



        ''' Adjust lengths '''

        qrs_i_raw = qrs_i_raw[:Beat_C1]
        qrs_amp_raw = qrs_amp_raw[:Beat_C1]
        qrs_c = qrs_c[:Beat_C+1]
        qrs_i = qrs_i[:Beat_C+1]
        
        return qrs_i_raw
    

def smoother(signal=None, kernel='boxzen', size=10, mirror=True, **kwargs):

        # check inputs
        if signal is None:
            raise TypeError("Please specify a signal to smooth.")
    
        length = len(signal)
    
        if isinstance(kernel, six.string_types):
            # check length
            if size > length:
                size = length - 1
    
            if size < 1:
                size = 1
    
            if kernel == 'boxzen':
                # hybrid method
                # 1st pass - boxcar kernel
                aux, _ = smoother(signal,
                                  kernel='boxcar',
                                  size=size,
                                  mirror=mirror)
    
                # 2nd pass - parzen kernel
                smoothed, _ = smoother(aux,
                                       kernel='parzen',
                                       size=size,
                                       mirror=mirror)
    
    #            params = {'kernel': kernel, 'size': size, 'mirror': mirror}
    
                return smoothed
    
            elif kernel == 'median':
                # median filter
                if size % 2 == 0:
                    raise ValueError(
                        "When the kernel is 'median', size must be odd.")
    
                smoothed = sig.medfilt(signal, kernel_size=size)
    
    #            params = {'kernel': kernel, 'size': size, 'mirror': mirror}
    
                return smoothed
    
            else:
                win = _get_window(kernel, size, **kwargs)
    
        elif isinstance(kernel, np.ndarray):
            win = kernel
            size = len(win)
    
            # check length
            if size > length:
                raise ValueError("Kernel size is bigger than signal length.")
    
            if size < 1:
                raise ValueError("Kernel size is smaller than 1.")
    
        else:
            raise TypeError("Unknown kernel type.")
    
        # convolve
        w = win / win.sum()
        if mirror:
            aux = np.concatenate(
                (signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))
            smoothed = np.convolve(w, aux, mode='same')
            smoothed = smoothed[size:-size]
        else:
            smoothed = np.convolve(w, signal, mode='same')
    
        # output
    #    params = {'kernel': kernel, 'size': size, 'mirror': mirror}
    #    params.update(kwargs)
    
        return smoothed
    
def _get_window(kernel, size, **kwargs):

    # mimics scipy.signal.get_window
    if kernel in ['blackman', 'black', 'blk']:
        winfunc = sig.blackman
    elif kernel in ['triangle', 'triang', 'tri']:
        winfunc = sig.triang
    elif kernel in ['hamming', 'hamm', 'ham']:
        winfunc = sig.hamming
    elif kernel in ['bartlett', 'bart', 'brt']:
        winfunc = sig.bartlett
    elif kernel in ['hanning', 'hann', 'han']:
        winfunc = sig.hann
    elif kernel in ['blackmanharris', 'blackharr', 'bkh']:
        winfunc = sig.blackmanharris
    elif kernel in ['parzen', 'parz', 'par']:
        winfunc = sig.parzen
    elif kernel in ['bohman', 'bman', 'bmn']:
        winfunc = sig.bohman
    elif kernel in ['nuttall', 'nutl', 'nut']:
        winfunc = sig.nuttall
    elif kernel in ['barthann', 'brthan', 'bth']:
        winfunc = sig.barthann
    elif kernel in ['flattop', 'flat', 'flt']:
        winfunc = sig.flattop
    elif kernel in ['kaiser', 'ksr']:
        winfunc = sig.kaiser
    elif kernel in ['gaussian', 'gauss', 'gss']:
        winfunc = sig.gaussian
    elif kernel in ['general gaussian', 'general_gaussian', 'general gauss',
                    'general_gauss', 'ggs']:
        winfunc = sig.general_gaussian
    elif kernel in ['boxcar', 'box', 'ones', 'rect', 'rectangular']:
        winfunc = sig.boxcar
    elif kernel in ['slepian', 'slep', 'optimal', 'dpss', 'dss']:
        winfunc = sig.slepian
    elif kernel in ['cosine', 'halfcosine']:
        winfunc = sig.cosine
    elif kernel in ['chebwin', 'cheb']:
        winfunc = sig.chebwin
    else:
        raise ValueError("Unknown window type.")

    try:
        window = winfunc(size, **kwargs)
    except TypeError as e:
        raise TypeError("Invalid window arguments: %s." % e)

    return window
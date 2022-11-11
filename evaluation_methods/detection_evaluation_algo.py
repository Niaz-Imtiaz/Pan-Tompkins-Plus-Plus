# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 28 16:17:44 2022

@author: Niaz
"""
import numpy as np

class Detection_evaluation():

    """

        Inputs
        ----------
         ecg : raw ecg vector signal 1d signal
         fs : sampling frequency e.g. 200Hz, 400Hz etc

        Outputs
        -------
        qrs_amp_raw : amplitude of R waves amplitudes
        qrs_i_raw : index of R waves
        delay : number of samples which the signal is delayed due to the filtering

    """
    def evaluate_qrs_detector(self, test, annotation, tol=0):
    
        test = np.unique(test)
        reference = np.unique(annotation)
    
        TP = 0
        tmp_reference=reference
        for anno_value in test:
            test_range = np.arange(anno_value-tol, anno_value+1+tol)
            in1d = np.in1d(tmp_reference, test_range).nonzero()[0]
            if in1d.size!=0:
                TP = TP + 1
                tmp_reference=np.delete(tmp_reference,in1d)
    
        FP = len(test)-TP
        FN = len(reference)-TP 

        return TP, FP, FN
    
    def evaluate_hr_detector(self, test, annotation, tol=0):
    
        test = np.unique(test)
        reference = np.unique(annotation)
    
        TP = 0

        for anno_value in test:
            test_range = np.arange(anno_value-tol, anno_value+1+tol)
            in1d = np.in1d(test_range, reference)
            if np.any(in1d):
                TP = TP + 1
    
        FP = len(test)-TP
        FN = len(reference)-TP 
        
        mean = 0
        std = 0

        return mean, std

    def evaluate_detection_metrics(self, TP, FP, FN):# Confusion matrix
        sensitivity = TP/(TP+FN)*100.0
        ppv = TP/(TP+FP)*100.0
        f1 = (2*TP)/((2*TP)+FP+FN)*100.0
    
        return sensitivity, ppv, f1

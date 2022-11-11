# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 28 11:32:20 2022

@author: Niaz
"""
import numpy as np
from datetime import datetime


def sort_MIT_annotations(ann):
    beat_labels = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    in_beat_labels = np.in1d(ann.symbol, beat_labels) #Check the the annotation of orfiginal signal matches any of the beat_labels. 
														#If matches take those symbols													
    sorted_anno = ann.sample[in_beat_labels]
    sorted_anno = np.unique(sorted_anno)

    return sorted_anno


def evaluate_detector(test, annotation, tol=0):

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

    return TP, FP, FN


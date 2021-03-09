from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convole(image,K):
	(iH,iW)=image.shape[:2]
	(kH,kW)=K.shape[:2]
	
	pad=(kW-1)//2
	image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_RELICATE)
	output=np.zeros((iH,iW),dtype='float')

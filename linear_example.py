import numpy as np
import cv2
import matplotlib.pyplot as plt

labels=['dog','cat','deer']
np.random.seed(1)

W=np.random.randn(3,3072)
b=np.random.randn(3)

orig=cv2.imread('beagle.png')
image=cv2.resize(orig,(32,32)).flatten()

print(image.shape)
print(W.shape)
#plt.plot(W)
#plt.show()
scores=W.dot(image)+b

print(zip(labels,scores))
for (label,score) in zip(labels,scores)):
	

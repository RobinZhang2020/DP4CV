from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

print('hello sgd')

ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,help='path to input dataset')
args=vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths=list(paths.list_images(args['dataset']))
#print(imagePaths)
sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])
(data,labels)=sdl.load(imagePaths,verbose=5)
#print(data.shape)
data=data.reshape((data.shape[0],3072))
#print(data)
#print(data.nbytes)
print('[INFO] features matrix: {:.1f}MB'.format(data.nbytes/(1024*1000.0)))

le=LabelEncoder()
labels=le.fit_transform(labels)
#print(len(labels))

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25,random_state=5)

for r in(None,'l1','l2'):
	#print(trainX.shape,testX.shape,trainY.shape,testY.shape)
	print("[INFO] training model with '{}'penalty".format(r))
	model=SGDClassifier(loss='log',penalty=r,max_iter=10,learning_rate='constant',eta0=0.01,random_state=42)
	model.fit(trainX,trainY)
	
	acc=model.score(testX,testY)
	print('[INFO] {} penalty accuracy: {:.2f}%'.format(r,acc*100))

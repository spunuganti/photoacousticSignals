
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import sys, getopt, os
import pdb
from random import shuffle
import logging
import cv2
#For transformation
import scipy.misc
import scipy.ndimage.interpolation
import random
import scipy.io as sio

os.environ['CUDA_VISIBLE_DEVICES']='3' #thin6 reserve a GPU

FolderName = 'D:\Researchvision_pytorch\Project'
SubFolder = '\ActiveSource_PZT1'
DataDivision = 100
use_gpu = torch.cuda.is_available()
NumEpochs = 500
BatchSizeTest = 50 #one at a time
LogFile = 'Log.txt'
learning_rate = 1e-6

#TODO
NumData = 0
BatchSize = 0

def main():
	##Load weight file
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hg:d", ["help", "load=","save="])
	except getopt.GetoptError as err:
	    # print help information and exit:
	    # print str(err)  # will print something like "option -a not recognized"
	    sys.exit(2)
	output = None
	verbose = False

	for opt, arg in opts:
	    if opt == "-v":
	        verbose = True
	    elif opt in ("-h", "--help"):
	        #print ()
	        sys.exit()
	    elif opt in ("-load", "--load"):
	        weightsInFile = arg

	    elif opt in ("-save", "--save"):
	        weightsOutFile = arg
	    else:
	        assert False, "unhandled option"


	#Log to file
	flog = open(LogFile, 'w')
	#load the text list        
	
	TrainList = []
	TestList = []


	#the model
	if use_gpu:
		model = Net().cuda()
		loss_fn = nn.BCELoss().cuda()
	else:
		model = Net()
		loss_fn = nn.BCELoss()

	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

	if 'weightsInFile' in locals(): #Load the existing weight
		model.load_state_dict(torch.load(weightsInFile))
		#Load test set
		correct = 0
		total = 0

		for num in range(NumTestDataSet):
			X1, X2, Labelset = Mydataloader(TestList, BatchSizeTest,num, ranListTest,0)
			predicted, correctT = TestModel(X1, X2, Labelset, optimizer, model)
			total += BatchSizeTest
			correct+=correctT
			#correct += (np.transpose(predicted.numpy()) == Labelset).sum()
		TopAc = correct / total
		#print('Accuracy of the saved network on the test images: {:.3f} %'.format(100 * TopAc))
		flog.write('Accuracy : {:.3f} % \n'.format(100 * TopAc))
			#logging.info('Accuracy of the saved network on the test images: {:.3f} %'.format(100 * TopAc))
		bestModelW = model.state_dict()
	else: 
		#Begin
		TopAc = 0

	for Epoch in range(NumEpochs): #Iterate with number of Epoches
		#Shuffle the list for every Epoch
		#Not updating with better one
		ranList=list(range(NumData))
		shuffle(ranList)

		lossAll = 0

		for num in range(DataDivision):
			X1, X2, Labelset = Mydataloader(TrainList, BatchSize,num, ranList,DataAug) #Aug for test data
			model, loss, correctT = TrainModel(X1, X2, Labelset,loss_fn,optimizer,model)
			lossAll += loss.data.cpu().numpy()[0]
		#print('Epoch{:03d} - Loss on the training batch images: {:.6f}'.format(Epoch+1, lossAll/num))
		flog.write('Epoch{:03d} - Loss : {:.6f}'.format(Epoch+1, lossAll/num))

		#Test 
		#Load train set
		NumTestData = len(TrainList)
		NumTestDataSet = int(NumTestData/BatchSizeTest)
		ranListTest=list(range(NumTestData)) # No shffle
		correct = 0
		total = 0

		for num in range(NumTestDataSet):
			X1, X2, Labelset = Mydataloader(TrainList, BatchSizeTest,num, ranListTest,0)
			predicted, correctT = TestModel(X1, X2, Labelset, optimizer, model)
			total += BatchSizeTest
			correct+=correctT
		Accuracy = correct / total
		#print('Epoch{:03d} - Accuracy on the train images: {:.3f} %'.format(Epoch+1, 100 * Accuracy))
		flog.write(' Epoch{:03d} - TrainAcc : {:.3f} % \n'.format(Epoch+1, 100 * Accuracy))

		#Test
		#Load test set
		NumTestData = len(TestList)
		NumTestDataSet = int(NumTestData/BatchSizeTest)
		ranListTest=list(range(NumTestData)) # No shffle
		correct = 0
		total = 0


		for num in range(NumTestDataSet):
			X1, X2, Labelset = Mydataloader(TestList, BatchSizeTest,num, ranListTest,0)
			predicted, correctT = TestModel(X1, X2, Labelset, optimizer, model)
			total += BatchSizeTest
			correct+=correctT

		Accuracy = correct / total
		#print('Epoch{:03d} - Accuracy on the test images: {:.3f} %'.format(Epoch+1, 100 * Accuracy))
		flog.write(' Epoch{:03d} - TestAcc : {:.3f} % \n'.format(Epoch+1, 100 * Accuracy))
		#logging.info('Epoch{:03d} - Accuracy on the test images: {:.3f} %'.format(Epoch+1, 100 * Accuracy))

		if Accuracy >= TopAc:
			#save to path when 
			bestModelW = model.state_dict()
			TopAc = Accuracy

			if 'weightsOutFile' in locals():
				torch.save(bestModelW, weightsOutFile)
			else:
				torch.save(bestModelW, 'WeightNoName')


def TestModel(X1, X2, Labelset,optimizer, model):


	X1 = torch.from_numpy(X1)
	X2 = torch.from_numpy(X2)
	Labelset = torch.from_numpy(Labelset)

		#wrap in variable
	if use_gpu:
		X1 = Variable(X1.cuda(),volatile=True).float()
		X2 = Variable(X2.cuda(),volatile=True).float()
		Labelset = Variable(Labelset.cuda(),volatile=True).float()
	else:
		X1 = Variable(X1,volatile=True).float()
		X2 = Variable(X2,volatile=True).float()
		Labelset = Variable(Labelset,volatile=True).float()

	optimizer.zero_grad()
	# forward + backward + optimize

	Output = model(X1,X2)
	#Binary prediction
	predicted = torch.round(Output.data)
	correctT = torch.sum(predicted == Labelset.data)

	return predicted, correctT


def TrainModel(X1, X2, Labelset,loss_fn,optimizer,model):

	# Data type conversion
	X1 = torch.from_numpy(X1)
	X2 = torch.from_numpy(X2)
	Labelset = torch.from_numpy(Labelset)

		#wrap in variable
	if use_gpu:
		X1 = Variable(X1.cuda()).float()
		X2 = Variable(X2.cuda()).float()
		Labelset = Variable(Labelset.cuda()).float()
	else:
		X1 = Variable(X1).float()
		X2 = Variable(X2).float()
		Labelset = Variable(Labelset).float()

	optimizer.zero_grad()

	Output = model(X1,X2)
	loss =loss_fn(Output, Labelset)
	loss.backward()
	optimizer.step()

	#Test
	predicted = torch.round(Output.data)
	correctT = torch.sum(predicted == Labelset.data)


	return model, loss, correctT


class Net(nn.Module):

	def __init__(self):
	    super(Net, self).__init__()
	    
	    # kernel
	    self.conv1 = nn.Conv2d(3, 64, 5,stride=(1,1), padding=2)
	    self.conv2 = nn.Conv2d(64, 128, 5,stride=(1,1), padding=2)
	    self.conv3 = nn.Conv2d(128, 256, 5,stride=(1,1), padding=2)
	    self.conv4 = nn.Conv2d(256, 512, 5,stride=(1,1), padding=2)
	    # an affine operation: y = Wx + b
	    self.fc1 = nn.Linear(16 * 16 * 512, 1024)
	    self.fc2 = nn.Linear(2048, 1)

	    self.norm1=nn.BatchNorm2d(64)
	    self.norm2=nn.BatchNorm2d(128)
	    self.norm3=nn.BatchNorm2d(256)
	    self.norm4=nn.BatchNorm2d(512)
	    self.norm5=nn.BatchNorm1d(1024)

	    self.sig = nn.Sigmoid()



	def forward(self, x, x2):
		# pytorch tutorial example as a skeleton
		x = F.max_pool2d(self.norm1(F.relu(self.conv1(x))), (2, 2)) #1~4
		x = F.max_pool2d(self.norm2(F.relu(self.conv2(x))), (2, 2)) #5~8
		x = F.max_pool2d(self.norm3(F.relu(self.conv3(x))), (2, 2)) #9~12
		x = self.norm4(F.relu(self.conv4(x))) #13~15
		x = x.view(-1, self.num_flat_features(x)) #16
		x = F.relu(self.fc1(x)) #17,18
		x = self.norm5(x)

		x2 = F.max_pool2d(self.norm1(F.relu(self.conv1(x2))), (2, 2)) #1~4
		x2 = F.max_pool2d(self.norm2(F.relu(self.conv2(x2))), (2, 2)) #5~8
		x2 = F.max_pool2d(self.norm3(F.relu(self.conv3(x2))), (2, 2)) #9~12
		x2 = self.norm4(F.relu(self.conv4(x2))) #13~15
		x2 = x2.view(-1, self.num_flat_features(x2)) #16
		x2 = F.relu(self.fc1(x2)) #17,18
		x2 = self.norm5(x2)

		#concatenates
		f12 = torch.cat((x,x2),1)
		output = self.sig(self.fc2(f12))

		return output

	def num_flat_features(self, x):
	    size = x.size()[1:] 
	    num_features = 1
	    for s in size:
	        num_features *= s
	    return num_features





def Mydataloader(Datalist, BatchSize, num, ranList, DataAug):
	##Load graphic file, n is nth databatch
	X1 = np.zeros((BatchSize,3,128,128))
	X2 = np.zeros((BatchSize,3,128,128))
	Labelset = np.zeros((BatchSize,1))

	BatchCount = 0

	#for t in Datalist[num*BatchSize:(num+1)*BatchSize]:
	for k in ranList[num*BatchSize:(num+1)*BatchSize]:
		t = Datalist[k]
		img1 = loadImages(FolderName+t[0])
		img2 = loadImages(FolderName+t[1])
		Label = int(t[2])
		#Rescaling for the model
		#Rimg1=cv2.resize(img1,(128,128)) #Cause error in GPU system at thin6 server
		#Rimg2=cv2.resize(img2,(128,128))
		Rimg1=scipy.misc.imresize(img1,(128,128))
		Rimg2=scipy.misc.imresize(img2,(128,128))
		#Convert to the format

		#Data Augmentation
		if DataAug:
			TransformImage(Rimg1)
			TransformImage(Rimg2)

		Rimg1=np.transpose(Rimg1,[2,0,1])
		Rimg2=np.transpose(Rimg2,[2,0,1])
		Rimg1.astype(np.float)
		Rimg2.astype(np.float)

		#normalize for training (weight)
		Rimg1 = normImage(Rimg1)
		Rimg2 = normImage(Rimg2)
		
		X1[BatchCount]=Rimg1
		X2[BatchCount]=Rimg2
		Labelset[BatchCount]=Label

		BatchCount += 1

	return X1, X2, Labelset

def normImage(Rimg1):
	Rimg1 =Rimg1-128
	Rimg1 =Rimg1/128
	return Rimg1



if __name__ == "__main__":
    main()

import struct
import math
from itertools import count
import time
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import queue
import librosa
import os
import scipy
import soundfile as sf
import pickle
import random



PICKLEROOT = "/Users/gavin/Documents/Harvard Classes/VES 161/sonicdream/"
SOUNDROOT = "/Users/gavin/Movies/Audio RAW Sonic Dream/"

DATABASENAME = "selectedMFCCchunks"


filenameKey = {}
# mfccChunkPoints = {}


filenames = []
os.chdir(PICKLEROOT[:-1])
for root, dirs, files in os.walk(".", topdown = False):
   for name in files:
      filenames.append(os.path.join(root, name))
   for name in dirs:
      filenames.append(os.path.join(root, name))

filenames = [name for name in filenames if "MFCCChunk" in name]

ShortenedMFCCList = []

ind = 0
totalLength = 0
for file in filenames:
	with open(file, 'rb') as f:
		print("opened: " + file + "   " + str(ind) + "/" + str(len(filenames)))
		npzfileobject = np.load(f)
		print("names in archive: " + str(npzfileobject.files))
		originalFilename = ''.join(npzfileobject["arr_2"].tolist())
		mfccChunks = npzfileobject["arr_0"]
		mfccChunkPoints = npzfileobject["arr_1"]
		print("sucessfully loaded")

		numToSelect = int(mfccChunkPoints.size ** 0.5)
		totalLength += numToSelect

		indices = random.sample(range(len(mfccChunkPoints)), numToSelect)
		print("selected " + str(numToSelect) + " points")
		mfccs = [mfccChunks[i, :, :] for i in indices]
		print("mfccs have shape: " + str(mfccs[0].shape))
		mfccPoints= [mfccChunkPoints[i] for i in indices]

		ShortenedMFCCList.append((originalFilename, mfccs, mfccPoints))
		print("added to shortlist")
	ind += 1

print("total length is: " + str(totalLength))

with open(DATABASENAME, 'wb+') as f:
	pickle.dump(ShortenedMFCCList , f)







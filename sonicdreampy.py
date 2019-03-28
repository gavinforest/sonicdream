# from https://stackoverflow.com/questions/32157871/python-real-time-audio-mixing

import struct
import math
import pyaudio
from itertools import count
import time
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import threading
import queue
import librosa
import pickle
import soundfile as sf
import os

x = np.random.rand(3000,3000)
y = np.random.rand(3000)

pa = pyaudio.PyAudio()


#float because that's what the mac microphone needs
PLAYER_FORMAT = pyaudio.paFloat32
PLAYER_CHANNELS = 1
PLAYER_RATE = 48000

LISTENER_FORMAT = pyaudio.paFloat32
LISTENER_CHANNELS = 2
LISTENER_RATE = 48000

OUTPUT_BLOCK_TIME = 0.05
OUTPUT_FRAMES_PER_BLOCK = int(PLAYER_RATE*OUTPUT_BLOCK_TIME) #DEBUG might be an error here if player and listener rates different

LISTENED_DATA_VOLUME_MULTIPLIER = 3.0

PICKLEROOT = "/Users/gavin/Documents/Harvard Classes/VES 161/sonicdream/"
SOUNDROOT = "/Users/gavin/Movies/Audio RAW Sonic Dream/"

DATABASENAME = "selectedMFCCchunks"
mfccShortlistComplete = None
with open(DATABASENAME, 'rb') as f:
	mfccShortlistComplete = pickle.load(f)
print("read database")
totalLength = sum([len(chPs) for o,m,chPs in mfccShortlistComplete])

mfccs = np.zeros((totalLength, 5 * 64))
mfccChunkPoints = np.zeros(totalLength)
mfccfilenames = []
allfilenames = []
currOffset = 0
print("beginning to create mfccs list")
for ofilename, mfccList, mfccchunkpoints in mfccShortlistComplete:
	num = len(mfccchunkpoints)
	allfilenames.append(ofilename)
	for iabs,irel in enumerate(range(currOffset, currOffset + num)):
		mfccs[irel,: ] = mfccList[iabs].ravel() ###USING RAVEL, BECAUSE ACTUAL FREQUENCIES DONT MATTER
		mfccChunkPoints[irel] = mfccchunkpoints[iabs]
		mfccfilenames.append(ofilename)

	currOffset += num

print("created mfccs list")
print("beginning to read all files included in the list")

start = time.time()
soundfileDict = {}
os.chdir("/Users/gavin/Movies/Audio RAW Sonic Dream")
for i, filename in enumerate(allfilenames):
	print("reading file " + str(i) + "/" + str(len(allfilenames)))
	sound, filerate = sf.read(filename, dtype='float32')

	if sound.ndim > 1:
		soundim = np.argmax(sound.shape)
		if soundim == 0:
			sound = sound[:,0]
		else:
			sound = sound[0,:]
		# print("had to cut stereo down to mono. Remaining audio first 48000 squared sum: " + str(np.sum(sound[:48000] * sound[:48000])))
	soundfileDict[filename] = sound
	print("successfully read in " + str(sound.size) + " data points")
# for i in range(len(mfccfilenames)):

end = time.time()
print("READ ALL FILES " + str(end - start))

bufferLock = threading.Lock()
inputbuffer = queue.Queue()

mfccNorms =np.sum(mfccs * mfccs, axis = 1) + 0.000000001

prevsoundfilename = None
prevEndPoint = 0
iterationsSincePlayed = 0

def getBlendArray(listenedBlockData):
	global prevsoundfilename
	global prevEndPoint
	global iterationsSincePlayed

	listenedBlock = librosa.feature.mfcc(listenedBlockData, sr=LISTENER_RATE, n_mfcc=64)
	listenedBlock = listenedBlock.ravel()

	listenedBlockNorm = np.sum(listenedBlock * listenedBlock) + 0.000000001

	dotProds = np.sum(mfccs * listenedBlock, axis = 1)
	dotProds = dotProds
	angles = dotProds * np.reciprocal(mfccNorms) / listenedBlockNorm

	maxangleInd = np.argmax(angles)
	print("highest similarity is: " + str(maxangleInd))

	maxangleChunkPoint = int(mfccChunkPoints[maxangleInd])
	maxangleIndFilename = mfccfilenames[maxangleInd]
	# print("prevsoundfilename is " + str(prev))
	sound = None

	if prevsoundfilename is None:
		sound = soundfileDict[maxangleIndFilename][maxangleChunkPoint : (maxangleChunkPoint + OUTPUT_FRAMES_PER_BLOCK)]
		prevEndPoint = maxangleChunkPoint + OUTPUT_FRAMES_PER_BLOCK
		prevsoundfilename = maxangleIndFilename
		iterationsSincePlayed += 1


	elif prevsoundfilename == maxangleIndFilename or iterationsSincePlayed < 10:
		sound = soundfileDict[prevsoundfilename][prevEndPoint : prevEndPoint + OUTPUT_FRAMES_PER_BLOCK]
		if sound.size < OUTPUT_FRAMES_PER_BLOCK:
			sound = soundfileDict[maxangleIndFilename][maxangleChunkPoint : (maxangleChunkPoint + OUTPUT_FRAMES_PER_BLOCK)]
			prevEndPoint = maxangleChunkPoint + OUTPUT_FRAMES_PER_BLOCK
			prevsoundfilename = maxangleIndFilename
			print("switched constant because ending")
			iterationsSincePlayed = 0
		else:
			prevEndPoint += OUTPUT_FRAMES_PER_BLOCK
			print("playing constant")
			iterationsSincePlayed += 1

	else:
		sound = soundfileDict[maxangleIndFilename][maxangleChunkPoint : (maxangleChunkPoint + OUTPUT_FRAMES_PER_BLOCK)]
		prevEndPoint = maxangleChunkPoint + OUTPUT_FRAMES_PER_BLOCK
		prevsoundfilename = maxangleIndFilename
		print("SWITCHED")
		iterationsSincePlayed = 0

	return sound


def recordCallback(inputdata, frame_count, time_info, status):
	print("in callback")
	# bufferLock.acquire()
	print(inputbuffer)
	data = np.frombuffer(inputdata, dtype= np.float32)
	leftData = np.reshape(data, (OUTPUT_FRAMES_PER_BLOCK, 2))
	# print("input data shape: " + str(data.shape))
	inputbuffer.put(leftData[:,0])
	# bufferLock.release()
	return (inputdata, pyaudio.paContinue)


def sine_gen():
	time = 0
	format = "%df" % OUTPUT_FRAMES_PER_BLOCK
	voices = []
	voices.append(lambda sampletime: math.sin(sampletime * math.pi * 2 * 440.0))

	for frame in count():
		# block = np.zeros(OUTPUT_FRAMES_PER_BLOCK, dtype = 'float32')
		block = []
		for i in range(OUTPUT_FRAMES_PER_BLOCK):
			sampletime = time + (float(i) / OUTPUT_FRAMES_PER_BLOCK) * OUTPUT_BLOCK_TIME
			sample = sum(voice(sampletime) for voice in voices) / len(voices)
			block.append(sample)
		# z = x * y
		print("yielding")
		yield struct.pack(format, *block)

		time += OUTPUT_BLOCK_TIME
		if frame == 20:
			voices.append(
				lambda sampletime: math.sin(sampletime * math.pi * 2 * 880.0)
			)

player = pa.open(format=PLAYER_FORMAT, output_device_index=2,
	channels=PLAYER_CHANNELS, rate=PLAYER_RATE, output=1)
listener = pa.open(format=LISTENER_FORMAT, 
	channels= LISTENER_CHANNELS, rate = PLAYER_RATE, input = True, input_device_index = 0,
	stream_callback=recordCallback, frames_per_buffer = OUTPUT_FRAMES_PER_BLOCK)

# i = 0
# while True:
# 	format = "%df" % OUTPUT_FRAMES_PER_BLOCK

# 	listenedBlockData = inputbuffer.get() * LISTENED_DATA_VOLUME_MULTIPLIER

# 	toBlendData = getBlendArray(listenedBlockData)

# 	totalBlock = soundfileDict[mfccfilenames[659]][i * OUTPUT_FRAMES_PER_BLOCK : (i + 1) * OUTPUT_FRAMES_PER_BLOCK]
# 	packedBlock = struct.pack(format, *totalBlock)
# 	player.write(packedBlock)
# 	i += 1


start = time.time()
# for i, block in enumerate(sine_gen()):
i = 0
while True:
	print("prevsoundfilename is " + str(prevsoundfilename))
	print("on loop: " + str(i))
	print("time to reenter loop: " + str(time.time() - start))
	start = time.time()

	totalBlock = None
	# blockData = np.frombuffer(block, dtype= np.float32) * 0.1
	
	errored = False
	# bufferLock.acquire()
	# print("got bufferLock")
	listenedBlockData = inputbuffer.get() * LISTENED_DATA_VOLUME_MULTIPLIER

	toBlendData= getBlendArray(listenedBlockData)
	print("toBlendData type: " + str(type(toBlendData)))

	print("listened block data amplitude squared: " + str(np.sum(listenedBlockData * listenedBlockData)))
	print("generated data has amplitude squared: " + str(np.sum(toBlendData * toBlendData)))

	# bufferLock.release()
	# try:
	# 	listenedBlock = listener.read(OUTPUT_FRAMES_PER_BLOCK)
	# 	# listenedBlockData = np.frombuffer(listenedBlock, dtype=np.float32)[:OUTPUT_FRAMES_PER_BLOCK]
	# except:
	# 	errored = True
	# 	print("errored, soldiering on")

	if i > 0:
		totalBlock = toBlendData
		# plt.plot(listenedBlockData)
		# plt.show()
	elif i == 0:
		totalBlock = listenedBlockData
	format = "%df" % OUTPUT_FRAMES_PER_BLOCK

	totalBlock = totalBlock * 0.8

	packedBlock = struct.pack(format, *totalBlock)

	player.write(packedBlock)
	print("wrote to stream in " + str(time.time() - start))
	print("input buffer size is: " + str(inputbuffer.qsize()))
	start = time.time()
	i += 1
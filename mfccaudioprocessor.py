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
import os
import scipy
import soundfile as sf
import pickle

x = np.random.rand(3000,3000)
y = np.random.rand(3000)

pa = pyaudio.PyAudio()


#float because that's what the mac microphone needs
PLAYER_FORMAT = pyaudio.paFloat32
PLAYER_CHANNELS = 1
PLAYER_RATE = 48000

LISTENER_FORMAT = pyaudio.paFloat32
LISTENER_CHANNELS = 1
LISTENER_RATE = 48000

OUTPUT_BLOCK_TIME = 0.05
OUTPUT_FRAMES_PER_BLOCK = int(PLAYER_RATE*OUTPUT_BLOCK_TIME) #DEBUG might be an error here if player and listener rates different

PICKLEROOT = "/Users/gavin/Documents/Harvard Classes/VES 161/sonicdream/"

bufferLock = threading.Lock()
inputbuffer = queue.Queue()

filenameKey = {}
# mfccChunkPoints = {}


filenames = []
os.chdir("/Users/gavin/Movies/Audio RAW Sonic Dream")
for root, dirs, files in os.walk(".", topdown = False):
   for name in files:
      filenames.append(os.path.join(root, name))
   for name in dirs:
      filenames.append(os.path.join(root, name))

filenames = [name for name in filenames if ".wav" in name]
# print(filenames)
chunkInd = 0
for filename in filenames:
	print("opening: " + filename + " " + str(chunkInd) + " / " + str(len(filenames)))
	
	sound, filerate = sf.read(filename, dtype='float32')
	print("read file, filerate: " + str(filerate))

	#get only first track if dual channel
	if sound.ndim > 1:
		soundim = np.argmax(sound.shape)
		if soundim == 0:
			sound = sound[:,0]
		else:
			sound = sound[0,:]
		print("had to cut stereo down to mono. Remaining audio first 48000 squared sum: " + str(np.sum(sound[:48000] * sound[:48000])))

	# if sound.dtype == "int16":
	# 	sound = sound.astype(np.float32, order='C') / 32768.0
	print("file dtype: " + str(sound.dtype))

	mfccChunkPoints = []
	print("starting to chunk mfccs for: " + filename)
	start = time.time()
	for i in range(int(sound.size / OUTPUT_FRAMES_PER_BLOCK)-10):
		mychunk = librosa.feature.mfcc(sound[i * OUTPUT_FRAMES_PER_BLOCK : (i+1) * OUTPUT_FRAMES_PER_BLOCK], sr = filerate, n_mfcc=92).tolist()
		# print("chunked in" + filename + str(end-start))
		# print(mychunk.shape)
		mfccChunkPoints.append((mychunk, i * OUTPUT_FRAMES_PER_BLOCK))

	print("pickling...")
	picklename = str(chunkInd)
	with open(PICKLEROOT + str(chunkInd)+"MFCCChunk", "wb+") as f:
		pickle.dump((picklename, mfccChunkPoints[:]), f)

	end = time.time()
	print("chunked " + filename + str(end-start))
	chunkInd += 1

# def recordCallback(inputdata, frame_count, time_info, status):
# 	print("in callback")
# 	# bufferLock.acquire()
# 	print(inputbuffer)
# 	inputbuffer.put(np.frombuffer(inputdata, dtype= np.float32))
# 	# bufferLock.release()
# 	return (inputdata, pyaudio.paContinue)


# def sine_gen():
# 	time = 0
# 	format = "%df" % OUTPUT_FRAMES_PER_BLOCK
# 	voices = []
# 	voices.append(lambda sampletime: math.sin(sampletime * math.pi * 2 * 440.0))

# 	for frame in count():
# 		# block = np.zeros(OUTPUT_FRAMES_PER_BLOCK, dtype = 'float32')
# 		block = []
# 		for i in range(OUTPUT_FRAMES_PER_BLOCK):
# 			sampletime = time + (float(i) / OUTPUT_FRAMES_PER_BLOCK) * OUTPUT_BLOCK_TIME
# 			sample = sum(voice(sampletime) for voice in voices) / len(voices)
# 			block.append(sample)
# 		# z = x * y
# 		print("yielding")
# 		yield struct.pack(format, *block)

# 		time += OUTPUT_BLOCK_TIME
# 		if frame == 20:
# 			voices.append(
# 				lambda sampletime: math.sin(sampletime * math.pi * 2 * 880.0)
# 			)

# player = pa.open(format=PLAYER_FORMAT,
# 	channels=PLAYER_CHANNELS, rate=PLAYER_RATE, output=1)
# listener = pa.open(format=LISTENER_FORMAT, 
# 	channels= LISTENER_CHANNELS, rate = PLAYER_RATE, input = True,
# 	 stream_callback=recordCallback, frames_per_buffer = OUTPUT_FRAMES_PER_BLOCK)

# start = time.time()
# for i, block in enumerate(sine_gen()):
# 	print("on loop: " + str(i))
# 	print("time to reenter loop: " + str(time.time() - start))
# 	start = time.time()

# 	totalBlock = None
# 	blockData = np.frombuffer(block, dtype= np.float32)
	
# 	errored = False
# 	# bufferLock.acquire()
# 	# print("got bufferLock")
# 	listenedBlockData = inputbuffer.get()

# 	mfccs = librosa.feature.mfcc(listenedBlockData, sr=LISTENER_RATE, n_mfcc=64)
# 	print("computed mfccs of size: " + str(mfccs.shape))
# 	# bufferLock.release()
# 	# try:
# 	# 	listenedBlock = listener.read(OUTPUT_FRAMES_PER_BLOCK)
# 	# 	# listenedBlockData = np.frombuffer(listenedBlock, dtype=np.float32)[:OUTPUT_FRAMES_PER_BLOCK]
# 	# except:
# 	# 	errored = True
# 	# 	print("errored, soldiering on")

# 	if i > 0:
# 		totalBlock = blockData + listenedBlockData
# 		# plt.plot(listenedBlockData)
# 		# plt.show()
# 	elif i == 0:
# 		totalBlock = blockData
# 	format = "%df" % OUTPUT_FRAMES_PER_BLOCK

# 	packedBlock = struct.pack(format, *totalBlock)

# 	player.write(packedBlock)
# 	print("wrote to stream in " + str(time.time() - start))
# 	print("input buffer size is: " + str(inputbuffer.qsize()))
# 	start = time.time()
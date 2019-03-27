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

LISTENED_DATA_VOLUME_MULTIPLIER = 10.0
bufferLock = threading.Lock()
inputbuffer = queue.Queue()

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

start = time.time()
for i, block in enumerate(sine_gen()):
	print("on loop: " + str(i))
	print("time to reenter loop: " + str(time.time() - start))
	start = time.time()

	totalBlock = None
	blockData = np.frombuffer(block, dtype= np.float32) * 0.1
	
	errored = False
	# bufferLock.acquire()
	# print("got bufferLock")
	listenedBlockData = inputbuffer.get() * LISTENED_DATA_VOLUME_MULTIPLIER
	print("listened block data amplitude squared: " + str(np.sum(listenedBlockData * listenedBlockData)))
	print("generated data has amplitude squared: " + str(np.sum(blockData * blockData)))

	mfccs = librosa.feature.mfcc(listenedBlockData, sr=LISTENER_RATE, n_mfcc=64)
	print("computed mfccs of size: " + str(mfccs.shape))
	# bufferLock.release()
	# try:
	# 	listenedBlock = listener.read(OUTPUT_FRAMES_PER_BLOCK)
	# 	# listenedBlockData = np.frombuffer(listenedBlock, dtype=np.float32)[:OUTPUT_FRAMES_PER_BLOCK]
	# except:
	# 	errored = True
	# 	print("errored, soldiering on")

	if i > 0:
		totalBlock = blockData + listenedBlockData
		# plt.plot(listenedBlockData)
		# plt.show()
	elif i == 0:
		totalBlock = blockData
	format = "%df" % OUTPUT_FRAMES_PER_BLOCK

	totalBlock = totalBlock * 0.8

	packedBlock = struct.pack(format, *totalBlock)

	player.write(packedBlock)
	print("wrote to stream in " + str(time.time() - start))
	print("input buffer size is: " + str(inputbuffer.qsize()))
	start = time.time()
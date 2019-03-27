# from https://stackoverflow.com/questions/32157871/python-real-time-audio-mixing

import struct
import math
import pyaudio
from itertools import count
import time
import numpy as np 
x = np.random.rand(3000,3000)
y = np.random.rand(3000)

pa = pyaudio.PyAudio()

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 48000

OUTPUT_BLOCK_TIME = 0.05
OUTPUT_FRAMES_PER_BLOCK = int(RATE*OUTPUT_BLOCK_TIME)


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
		z = x * y
		print("yielding")
		yield struct.pack(format, *block)

		time += OUTPUT_BLOCK_TIME
		if frame == 20:
			voices.append(
				lambda sampletime: math.sin(sampletime * math.pi * 2 * 880.0)
			)

player = pa.open(format=FORMAT,
	channels=CHANNELS, rate=RATE, output=1)
start = time.time()
for i, block in enumerate(sine_gen()):
	print("time to reenter loop: " + str(time.time() - start))
	start = time.time()
	player.write(block)
	print("wrote to stream in " + str(time.time() - start))
	start = time.time()
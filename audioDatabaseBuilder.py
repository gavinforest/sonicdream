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
SOUNDROOT = "/Users/gavin/Movies/Audio RAW Sonic Dream/"


bufferLock = threading.Lock()
inputbuffer = queue.Queue()

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

for file in filenames:
	sourcefilename, mfcclists = 
import pyaudio
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
print("Num devices: " + str(numdevices))
for i in range(0, numdevices):
	inchannels = p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')
	outchannels = p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')
	if inchannels > 0:
		print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name') + "  channels - " + str(inchannels))
	elif outchannels > 0:
		print("Output Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name') + "  channels - " + str(outchannels))


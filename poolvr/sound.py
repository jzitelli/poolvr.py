import pkgutil
try:
    import sounddevice as sd
except ImportError as err:
    print('could not import package "sounddevice":\n%s' % err)
    sd = None
try:
    import soundfile as sf
except ImportError as err:
    print('could not import package "soundfile":\n%s' % err)
    sf = None


def open_output_stream(device=None, channels=2, samplerate=44100):
    return sd.OutputStream(device=device, channels=channels, samplerate=samplerate)

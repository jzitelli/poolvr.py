# import pkgutil
import os.path
import logging
import numpy as np


_logger = logging.getLogger(__name__)


try:
    import sounddevice as sd
    try:
        import soundfile as sf
    except ImportError as err:
        _logger.error('could not import soundfile:\n%s', err)
        sf = None
except ImportError as err:
    _logger.error('could not import sounddevice:\n%s', err)
    _logger.error('SOUND IS NOT AVAILABLE')
    sd = None


SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.path.pardir, 'sounds')


_initialized = False
ballBall_sound = None
ballBall_sound_fs = None
output_stream = None


def init_sound():
    global _initialized
    global ballBall_sound
    global ballBall_sound_fs
    if not _initialized:
        if sf:
            ballBall_sound, ballBall_sound_fs = sf.read(os.path.join(SOUNDS_DIR, 'ballBall.ogg'))
            ballBall_sound *= 0.6
            ballBall_sound = np.array(ballBall_sound, dtype=np.float32)
        if sd:
            # sd.default.clip_off = True
            # sd.default.dither_off = True
            # sd.default.samplerate = 44100
            # sd.default.never_drop_input = True
            output_stream = open_output_stream()
            output_stream.start()
        _initialized = True


def open_output_stream(device=None, channels=2, samplerate=44100):
    global output_stream
    if sd:
        output_stream = sd.OutputStream(dtype='float32')#device=device, samplerate=samplerate, latency='low')
                                        #clip_off=True, never_drop_input=True, dither_off=True, latency='low')
        return output_stream


def play_ball_ball_collision_sound(vol=1.0):
    if output_stream:
        output_stream.write(vol*ballBall_sound[:output_stream.write_available])
        #sd.play(vol * ballBall_sound, ballBall_sound_fs)


def list_sound_devices():
    if sd:
        _logger.info("""

#####################################################

                        listing sound devices...

#####################################################

                     """)
        devices = sd.query_devices() #kind='output')
                   #if device['max_output_channels'] > 0]
        if isinstance(devices, dict): # i.e. only one sound device found
            devices = [devices]
        for i, device in enumerate(devices):
            if device['max_output_channels'] == 0:
                continue
            print('device %2d: %s: %d channels' % (i, device['name'], device['max_output_channels']))


def set_output_sound_device(device):
    if sd:
        sd.default.device[1] = device
        # _logger.info('set output sound device to %d: %s', device, sd.query_devices()[device]['name'])

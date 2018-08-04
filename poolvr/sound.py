# import pkgutil
import os.path
import logging
import numpy as np


_logger = logging.getLogger(__name__)


try:
    import sounddevice as sd
except ImportError as err:
    sd = None
    _logger.error('could not import sounddevice:\n%s', err)
try:
    import soundfile as sf
except ImportError as err:
    sf = None
    _logger.error('could not import soundfile:\n%s', err)

if sf is None or sd is None:
    _logger.error('SOUND IS NOT AVAILABLE')
    _avail = False
else:
    _avail = True

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.path.pardir, 'sounds')


_initialized = False
ballBall_sound = None
ballBall_sound_fs = None
ballBall_temp = None
output_stream = None


def only_if_avail(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _avail
        if _avail:
            return func(*args, **kwargs)
    return wrapper


@only_if_avail
def init_sound():
    global _initialized
    global ballBall_sound
    global ballBall_sound_fs
    global ballBall_temp
    if not _initialized:
        ballBall_sound, ballBall_sound_fs = sf.read(os.path.join(SOUNDS_DIR, 'ballBall.ogg'))
        ballBall_sound = np.array(ballBall_sound, dtype=np.float32)
        ballBall_temp = ballBall_sound.copy()
        open_output_stream()
        _initialized = True


@only_if_avail
def open_output_stream(device=None, channels=2, samplerate=44100):
    global output_stream
    output_stream = sd.OutputStream(dtype='float32')
    # device=device, samplerate=samplerate, latency='low',
    # clip_off=True, dither_off=True, never_drop_input=True)
    output_stream.start()

# _n = 0
# _vol = 0.0
# _vols = []


@only_if_avail
def play_ball_ball_collision_sound(vol=1.0):
    # global _n
    # global _vol
    # global _vols
    if output_stream:
        n = output_stream.write_available
        # _n += n
        # q, r = (_n // len(ballBall_sound),
        #         _n - (_n // len(ballBall_sound)) * len(ballBall_sound))
        ballBall_temp[:n] = ballBall_sound[:n]
        ballBall_temp[:n] *= vol
        # r = min(r, n)
        # ballBall_temp[n-r:n] += _vol * ballBall_sound[r:n]
        output_stream.write(ballBall_temp[:n])
        # _vol = vol


@only_if_avail
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


@only_if_avail
def set_output_sound_device(device):
    if sd:
        sd.default.device[1] = device
        _logger.info('set output sound device to %d: %s', device, sd.query_devices()[device]['name'])

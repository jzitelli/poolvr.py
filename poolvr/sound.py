# import pkgutil
import os.path
import logging


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


def init_sound():
    global _initialized
    global ballBall_sound
    global ballBall_sound_fs
    if not _initialized:
        ballBall_sound, ballBall_sound_fs = sf.read(os.path.join(SOUNDS_DIR, 'ballBall.ogg'))
        ballBall_sound *= 0.2
        _initialized = True


def open_output_stream(device=None, channels=2, samplerate=44100):
    if sd:
        return sd.OutputStream(device=device, channels=channels, samplerate=samplerate)


def play_ball_ball_collision_sound(vol=1.0):
    if sd:
        sd.play(vol * ballBall_sound, ballBall_sound_fs)


def list_sound_devices():
    if sd:
        devices = sd.query_devices(kind='output')
        if isinstance(devices, dict):
            devices = [devices]
        for device in devices:
            print('%s: %d channels' % (device['name'], device['max_output_channels']))


def set_output_sound_device(device):
    if sd:
        sd.default.device[1] = device
        # _logger.info('set output sound device to %d: %s', device, sd.query_devices()[device]['name'])

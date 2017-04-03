# import pkgutil
import os.path
import logging
try:
    import soundfile as sf
except ImportError as err:
    print('could not import package "soundfile":\n%s' % err)
    sf = None
try:
    import sounddevice as sd
except ImportError as err:
    print('could not import package "sounddevice":\n%s' % err)
    sd = None


_logger = logging.getLogger(__name__)


SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.path.pardir, 'sounds')


_initialized = False


ballBall_sound = None
ballBall_sound_fs = None


def init():
    global _initialized
    if not _initialized:
        global ballBall_sound
        global ballBall_sound_fs
        ballBall_sound, ballBall_sound_fs = sf.read(os.path.join(SOUNDS_DIR, 'ballBall.ogg'))
        ballBall_sound *= 0.2
    _initialized = True


def open_output_stream(device=None, channels=2, samplerate=44100):
    return sd.OutputStream(device=device, channels=channels, samplerate=samplerate)


def play_ball_ball_collision_sound(vol=1.0):
    if _initialized:
        sd.play(vol * ballBall_sound, ballBall_sound_fs, device=14)

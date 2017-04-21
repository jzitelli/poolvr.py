# import pkgutil
import os.path
import logging


_logger = logging.getLogger(__name__)


SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.path.pardir, 'sounds')


_initialized = False

sd = None
sf = None

ballBall_sound = None
ballBall_sound_fs = None

def init_sound():
    global _initialized
    global sd
    global sf
    global ballBall_sound
    global ballBall_sound_fs
    if not _initialized:
        try:
            import sounddevice as _sd
            sd = _sd
        except ImportError as err:
            _logger.error('could not import sounddevice:\n%s' % err)
            _logger.error('SOUND IS NOT AVAILABLE')
            _initialized = True
            return
        import soundfile as _sf
        sf = _sf
        ballBall_sound, ballBall_sound_fs = sf.read(os.path.join(SOUNDS_DIR, 'ballBall.ogg'))
        ballBall_sound *= 0.2
        _initialized = True


def open_output_stream(device=None, channels=2, samplerate=44100):
    if sd:
        return sd.OutputStream(device=device, channels=channels, samplerate=samplerate)


def play_ball_ball_collision_sound(vol=1.0):
    if sd:
        sd.play(vol * ballBall_sound, ballBall_sound_fs, device=14)

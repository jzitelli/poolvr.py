# poolvr.py

VR pool simulator written in Python (using [pyopenvr](https://github.com/cmbruns/pyopenvr))

![screenshot](https://jzitelli.github.io/poolvr.py/images/screenshots/vrscreenshot.png)


### REQUIREMENTS:

- Python 3.5 or higher
- [glfw](https://github.com/FlorianRhiem/pyGLFW)
- [pyopengl](http://pyopengl.sourceforge.net)
- [numpy](http://www.numpy.org)
- [pillow](https://python-pillow.org)

#### Optional dependencies:

- [pyopenvr](https://github.com/cmbruns/pyopenvr)
  for VR
- [sounddevice](https://pypi.org/project/sounddevice)
  and [soundfile](https://github.com/bastibe/SoundFile)
  for sound

#### Developer dependencies:

- [pytest](https://www.pytest.org)
  and [matplotlib](https://matplotlib.org)



### INSTALLING poolvr.py:

1. Install the required dependencies.


2. Build and install the `poolvr` package:
```
cd <poolvr.py root dir>
python setup.py install
```



### STARTING `poolvr.py`:

To start `poolvr` in VR-mode, run from command-line:
```
poolvr
```

To run without VR:
```
poolvr --novr
```

To see all available command-line options:
```
poolvr -h
```



### RUNNING THE TESTS:

```
cd <poolvr.py root dir>/test
pytest
```

To see all available test command-line options:
```
pytest -h
```

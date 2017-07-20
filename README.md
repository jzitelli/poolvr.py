# poolvr.py

VR pool simulator written in Python (using [pyopenvr](https://github.com/cmbruns/pyopenvr))

![screenshot](https://jzitelli.github.io/poolvr.py/images/screenshots/Screenshot%202017-04-08%2003.25.27.png)

## REQUIREMENTS:

- Python 3.5 or higher
- cyglfw3
- pyopengl
- pyopenvr
- numpy
- pillow
- matplotlib
- python-sounddevice
- soundfile

Time-stepped physics using the [Open Dynamics Engine](https://github.com/jzitelli/ode) is also supported if the ODE library with Python bindings is installed.

You can probably install most of the required packages listed above using `pip`, e.g.
```
pip install pillow
```
Others, such as `cyglfw3` and `ode`, I found I had to build from source.


### INSTALLING `cyglfw3`:

First, you need to build or download the `glfw` library binary for your platform - the easiest way is to [download pre-compiled binaries
from the official `glfw` site: http://www.glfw.org/download.html ](http://www.glfw.org/download.html)

Then build the cyglfw3 package:
```
git clone https://github.com/jzitelli/cyglfw3.git
cd cyglfw3
python setup.py build_py build_ext install --include-dirs="{path to glfw include dir}" --library-dirs="{path to glfw dll dir}"
```


### INSTALLING `ode`:

`ode` is the Python package of bindings for the Open Dynamics Engine.  The library and bindings are built from the same source repository:
```
git clone https://github.com/jzitelli/ode.git
cd ode
```

## HOW TO INSTALL:

```
git clone https://github.com/jzitelli/poolvr.py.git
cd poolvr.py
python setup.py develop
```


## HOW TO START:

Installation will place a Python script `play_poolvr.py` into your Python environment's path.
If you are using a bash command-line, just enter:
```
play_poolvr.py
```

To run without VR:
```
play_poolvr.py --novr
```

For information on available command-line options and other help:
```
play_poolvr.py -h
```

## HOW TO RUN TESTS:

From your cloned *poolvr.py* repository root directory:
```
python scripts/run_poolvr_tests.py
```

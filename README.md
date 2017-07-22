# poolvr.py

VR pool simulator written in Python (using [pyopenvr](https://github.com/cmbruns/pyopenvr))

![screenshot](https://jzitelli.github.io/poolvr.py/images/screenshots/vrscreenshot.png)

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
I was not able to install `cyglfw3`, 'pyopenvr', or `ode` using `pip`, so I built them from source following the steps below:


### 1. INSTALLING `cyglfw3`:

First, you need to build or download the `glfw` library binary for your platform - the easiest way is to [download pre-compiled binaries
from the official `glfw` site: http://www.glfw.org/download.html ](http://www.glfw.org/download.html)

Then build the cyglfw3 package:
```
git clone https://github.com/jzitelli/cyglfw3.git
cd cyglfw3
python setup.py build_py build_ext install --include-dirs="{path to glfw include dir}" --library-dirs="{path to glfw dll dir}"
```


### 2. INSTALLING `pyopenvr`:

At the moment I am using my own fork of [https://github.com/cmbruns/pyopenvr](https://github.com/cmbruns/pyopenvr).  To clone and install:
```
git clone https://github.com/jzitelli/pyopenvr.git
cd pyopenvr
python setup.py install
```


### 3. INSTALLING `ode`:

`ode` is the Python package of bindings for the Open Dynamics Engine.  The library and bindings are built from the same source repository.  To clone the repo and generate a Visual Studio solution (`.sln`) for building the library:
```
git clone https://github.com/jzitelli/ode.git
cd ode
cd build
premake4.exe --only-shared --only-double --platform=x64 vs2010
```
Then open the generated solution in Visual Studio (probably ok to upgrade the solution to your version of VS if it asks you - I tested successfully with 2015 and 2017).
Compile a Release build for your target architecture (I believe this should match your version of Python, e.g. x64 or x86).
It should output a library `ode\lib\Release\ode.dll`.  You should add this location to your PATH environment variable or copy the file to a directory in your PATH.

To build the Python bindings, run from the Visual Studio Native Tools Command Line:
```
cd {directory where you cloned the repo}
cd bindings
cd python
python setup.py build_ext install
```
If installed successfully, from the Python interpreter you should be able to import the `ode` package, e.g.
```
import ode
print(ode.__file__) # <-- assuming Python 3 here
```
and see something like `...\Anaconda3\lib\site-packages\ode.cp36-win_amd64.pyd`.


## HOW TO INSTALL poolvr.py:

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

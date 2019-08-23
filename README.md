# poolvr.py

VR pool simulator written in Python (using [pyopenvr](https://github.com/cmbruns/pyopenvr))

![screenshot](https://jzitelli.github.io/poolvr.py/images/screenshots/vrscreenshot.png)


### REQUIREMENTS:

- Python 3.5 or higher
- [cyglfw3](https://github.com/adamlwgriffiths/cyglfw3)
- [pyopengl](http://pyopengl.sourceforge.net)
- [numpy](http://www.numpy.org)
- [pillow](https://python-pillow.org)

#### Optional dependencies:

- [pyopenvr](https://github.com/cmbruns/pyopenvr)
  for VR
- [ode](https://ode):
  Python-bindings to the [Open Dynamics Engine](https://ode)
  which provides a time-stepped pool physics simulation
  (rather than the internal event-based simulation)
- [sounddevice](https://pypi.org/project/sounddevice)
  and [soundfile](https://github.com/bastibe/SoundFile)
  for sound

#### Developer dependencies:

- [pytest](https://www.pytest.org)
  and [matplotlib](https://matplotlib.org)



### INSTALLING poolvr.py:

1. Install the required dependencies.
   You can probably install most of the required packages via `pip`
   with the following exceptions:

   `cyflw3`: If `pip install cyglfw3` fails,
   you may try building the package yourself::
   
     1. Build or download the `glfw` library binary for your platform:
     The easiest way is to download pre-compiled binaries
     from the official `glfw` site:
     [http://www.glfw.org/download.html]

     2. Clone and build the cyglfw3 package:
     ```
     git clone https://github.com/jzitelli/cyglfw3.git
     cd cyglfw3
     python setup.py build_py build_ext \
     --include-dirs=<path to glfw include dir> --library-dirs=<path to glfw dll dir>
     python setup.py install
     ```

   `ode`: If `pip install ode` fails,
   you may try building the package yourself::

     1. Clone and generate a Visual Studio solution (`.sln`)
     for building the library:
     ```
     git clone https://github.com/jzitelli/ode.git
     cd ode/build
     premake4.exe --only-shared --only-double --platform=x64 vs2010
     ```
     
     2. Open the generated solution in Visual Studio
     and follow any upgrade suggestions that your version
     of Visual Studio makes.

     3. Compile a Release build for your target architecture
	   (I believe this should match your version of Python, e.g. x64 or x86).

     4. Copy the built library `ode.dll` to a location in your PATH.

     5. Build the Python bindings by running from 
     the Visual Studio Native Tools command-line:
     ```
     cd <ode root directory>/bindings/python
     python setup.py build_ext install
     ```

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

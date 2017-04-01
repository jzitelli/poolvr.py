# poolvr.py

VR pool simulator written in Python (using pyopenvr)


## REQUIREMENTS:

- Python 3.5 or higher
- cyglfw3
- pyopengl
- pyopenvr
- numpy
- pillow
- matplotlib

Time-stepped physics using the [Open Dynamics Engine](https://github.com/jzitelli/ode) is also supported if the ODE library with Python bindings is installed.

You can probably install most of the required packages listed above using `pip`, e.g.
```
pip install pillow
```
Others, such as `cyglfw3` and `ode`, I found I had to build from source.


## HOW TO INSTALL:

```
git clone https://github.com/jzitelli/poolvr.py.git
cd poolvr.py
python setup.py develop
```


## HOW TO START:

Installation will place a Python script `play_poolvr.py` into your Python environment's path.
Run it from the command-line:
```
python play_poolvr.py
```

To run without VR:
```
python play_poolvr.py --novr
```

For information on command-line options and other help:
```
python play_poolvr.py -h
```

## HOW TO RUN TESTS:

From the *poolvr.py* repo root directory:
```
python scripts/run_poolvr_tests.py
```

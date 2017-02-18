# poolvr.py

VR pool simulator written in Python (using pyopenvr)


## REQUIREMENTS:

- Python 3.5 or higher
- cyglfw3
- pyopengl
- pyopenvr
- numpy
- pillow

If you want to run the tests, the following are also required:

- matplotlib


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

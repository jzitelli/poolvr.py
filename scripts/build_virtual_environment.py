#!python
import os.path
from time import time
from subprocess import run, PIPE
from sys import stdout


_here = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.join(_here, os.path.pardir, 'venv36')
venv_abspath = os.path.abspath(venv_path)
activate_path = os.path.join('venv36', 'Scripts', 'activate')


def main():
    print('''
building virtual environment in %s...
''' % venv_abspath)
    stdout.flush()
    t0 = time()
    run(['virtualenv', '-v', venv_abspath], stdout=PIPE, stderr=PIPE)
    t1 = time()
    print('''
...done building virtual environment (took %s seconds)

installing requirements...
''' % (t1 - t0))

    t2 = time()
    run('source %s && pip install -r requirements.txt' % activate_path,
        cwd=os.path.join(_here, os.path.pardir),
        shell=True, stdout=PIPE, stderr=PIPE)
    t3 = time()
    print('''
...done installing requirements (took %s seconds)
''' % (t3 - t2))


if __name__ == "__main__":
    main()

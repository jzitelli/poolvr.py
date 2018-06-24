#!python
import os.path
from time import time
from subprocess import run, PIPE
from sys import stdout
import logging


_here = os.path.dirname(os.path.abspath(__file__))
_logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    venv_path = os.path.join(_here, os.path.pardir, 'venv36')
    venv_abspath = os.path.abspath(venv_path)
    activate_path = os.path.join('venv36', 'Scripts', 'activate')
    _logger.info('''Building virtual environment in %s...''', venv_abspath)
    stdout.flush()
    t0 = time()
    command = ['virtualenv', '-v', venv_abspath]
    _logger.info('\n'.join(l.lstrip() for l in '''Creating virtual environment...

    $ %s
    '''.split('\n')), ' '.join(command))
    run(command, stdout=PIPE, stderr=PIPE)
    t1 = time()
    command = 'source %s && pip install -r requirements.txt' % activate_path
    _logger.info('\n'.join(l.lstrip() for l in '''...done creating virtual environment (took %s seconds).

    Installing requirements into virtual environment...

    $ %s
    '''.split('\n')), (t1-t0), command)
    t2 = time()
    run(command,
        cwd=os.path.join(_here, os.path.pardir),
        shell=True, stdout=PIPE, stderr=PIPE)
    t3 = time()
    _logger.info('\n'.join(l.lstrip() for l in '''...done installing requirements (took %s seconds).

    ...done building virtual environment (took %s seconds).'''.split('\n')), (t3-t2), (t3-t0))


if __name__ == "__main__":
    main()

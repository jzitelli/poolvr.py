#!/bin/bash
version_suffix=$(python -c "from sys import version_info as vi; print('%s%s' % (vi.major, vi.minor))")
set -x
python -m venv "venv$version_suffix"
set +x
. "venv$version_suffix"/Scripts/activate
set -x
pip install -r requirements.txt

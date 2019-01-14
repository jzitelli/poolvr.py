#!/bin/bash
py3=$(python3 -c 'print(1)')
if [[ -z ${py3} ]]; then
    pycmd="python"
else
    pycmd="python3"
fi
version_suffix=$($pycmd -c "from sys import version_info as vi; print('%s%s' % (vi.major, vi.minor))")

project_root="${0%/*}/.."
venv_path="${project_root}/venv${version_suffix}"
if [[ ! -d "${venv_path}" ]]; then
    echo "creating virtual environment for python ${version_suffix} in ${venv_path}..."
    # set -x
    $pycmd -m venv "${venv_path}"
    # set +x
fi

echo "activating virtual environment in ${venv_path}..."
if [[ "$OS" == "Windows_NT" ]]; then 
    source "./${venv_path}/Scripts/activate"
else
    source "./${venv_path}/bin/activate"
fi
# set -x
echo "installing requirements from requirements.txt..."
pip install -r requirements.txt
echo "deactivating virtual environment..."
deactivate

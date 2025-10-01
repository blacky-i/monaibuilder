#! /bin/bash

# Copyright (c) xxxx-2023 MONAI Consortium
# Copyright (c) 2023- Chernenkiy Ivan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# script for running all tests
set -e

# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]] # stdout is a terminal
then
    separator=$'--------------------------------------------------------------------------------\n'
    blue="$(tput bold; tput setaf 4)"
    green="$(tput bold; tput setaf 2)"
    red="$(tput bold; tput setaf 1)"
    noColor="$(tput sgr0)"
fi

# configuration values
doDryRun=false
doBlackFormat=false
doBlackFix=false
doIsortFormat=false
doIsortFix=false
doFlake8Format=false
doPylintFormat=false
doCopyRight=false
doPytypeFormat=false
doMypyFormat=false

NUM_PARALLEL=1
PY_EXE=${PY_EXE:-$(which python)}

function print_usage {
    echo "runtests.sh [--codeformat] [--autofix] "
    echo "            [--dryrun] [-j number]"
    echo "            [--help]"
    echo ""
    echo "MONAI unit testing utilities."
    echo ""
    echo "Examples:"
    echo "./runtests.sh -f                      # run coding style and static type checking."
    echo "./runtests.sh --autofix               # run automatic code formatting using \"isort\" and \"black\"."
    echo ""
    echo "Python type check options:"
    echo "    -j, --jobs        : number of parallel jobs to run \"pytype\" (default $NUM_PARALLEL)"
    echo ""
    echo "Misc. options:"
    echo "    --dryrun          : display the commands to the screen without running"
    echo "    -f, --codeformat  : shorthand to run all code style and static analysis tests"
    echo "    -h, --help        : show this help message and exit"
    echo ""
    echo "${separator}For bug reports and feature requests, please file an issue at:"
    echo "    https://github.com/Project-MONAI/MONAI/issues/new/choose"
    echo ""
    echo "To choose an alternative python executable, set the environmental variable, \"PY_EXE\"."
    exit 1
}


function print_style_fail_msg() {
    echo "${red}Check failed!${noColor}"
    echo "Please run auto style fixes: ${green}./runtests.sh --autofix${noColor}"
}


function print_error_msg() {
    echo "${red}Error: $1.${noColor}"
    echo ""
}


# Обработка аргументов
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --dryrun)
            doDryRun=true
        ;;
        -f|--codeformat)
            doBlackFormat=true
            doIsortFormat=true
            doFlake8Format=true
            doPylintFormat=true
            doCopyRight=true
            doPytypeFormat=true
            doMypyFormat=true
        ;;
        --autofix)
            doIsortFix=true
            doBlackFix=true
            doIsortFormat=true
            doBlackFormat=true
            doCopyRight=true
        ;;
    esac
    shift
done   

# Домашняя директория

HOMEDIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$HOMEDIR"

cmdPrefix=""

if [ $doDryRun = true ]
then
    echo "${separator}${blue}dryrun${noColor}"

    # commands are echoed instead of ran
    cmdPrefix="dryrun "
    function dryrun { echo "    " "$@"; }
fi

if [ $doCopyRight = true ]
then
    # check copyright headers
    copyright_bad=0
    copyright_all=0
    while read -r fname; do
        copyright_all=$((copyright_all + 1))
        if ! grep "http://www.apache.org/licenses/LICENSE-2.0" "$fname" > /dev/null; then
            print_error_msg "Missing the license header in file: $fname"
            copyright_bad=$((copyright_bad + 1))
        fi
    done <<< "$(find "$(pwd)/monai" "$(pwd)/tests" -type f \
        ! -wholename "*_version.py" -and -name "*.py" -or -name "*.cpp" -or -name "*.cu" -or -name "*.h")"
    if [[ ${copyright_bad} -eq 0 ]];
    then
        echo "${green}Source code copyright headers checked ($copyright_all).${noColor}"
    else
        echo "Please add the licensing header to the file ($copyright_bad of $copyright_all files)."
        echo "  See also: https://github.com/Project-MONAI/MONAI/blob/dev/CONTRIBUTING.md#checking-the-coding-style"
        echo ""
        exit 1
    fi
fi

if [ $doIsortFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    if [ $doIsortFix = true ]
    then
        echo "${separator}${blue}isort-fix${noColor}"
    else
        echo "${separator}${blue}isort${noColor}"
    fi

    if [ $doIsortFix = true ]
    then
        ${cmdPrefix}${PY_EXE} -m isort "$(pwd)"
    else
        ${cmdPrefix}${PY_EXE} -m isort --check "$(pwd)"
    fi

    isort_status=$?
    if [ ${isort_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${isort_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doBlackFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    if [ $doBlackFix = true ]
    then
        echo "${separator}${blue}black-fix${noColor}"
    else
        echo "${separator}${blue}black${noColor}"
    fi

    ${cmdPrefix}${PY_EXE} -m black --version

    if [ $doBlackFix = true ]
    then
        ${cmdPrefix}${PY_EXE} -m black --skip-magic-trailing-comma "$(pwd)"
    else
        ${cmdPrefix}${PY_EXE} -m black --skip-magic-trailing-comma --check "$(pwd)"
    fi

    black_status=$?
    if [ ${black_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${black_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi

if [ $doFlake8Format = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}flake8${noColor}"

    ${cmdPrefix}${PY_EXE} -m flake8 --version

    ${cmdPrefix}${PY_EXE} -m flake8 "$(pwd)" --count --statistics

    flake8_status=$?
    if [ ${flake8_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${flake8_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi

if [ $doPylintFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}pylint${noColor}"

    # ensure that the necessary packages for code format testing are installed
    ${cmdPrefix}${PY_EXE} -m pylint --version

    ignore_codes="C,R,W,E1101,E1102,E0601,E1130,E1123,E0102,E1120,E1137,E1136"
    ${cmdPrefix}${PY_EXE} -m pylint monaibuilder tests --disable=$ignore_codes -j $NUM_PARALLEL
    pylint_status=$?

    if [ ${pylint_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${pylint_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi

if [ $doPytypeFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}pytype${noColor}"

    pytype_ver=$(${cmdPrefix}${PY_EXE} -m pytype --version)
    if [[ "$OSTYPE" == "darwin"* && "$pytype_ver" == "2021."* ]]; then
        echo "${red}pytype not working on macOS 2021 (https://github.com/Project-MONAI/MONAI/issues/2391). Please upgrade to 2022*.${noColor}"
        exit 1
    else
        ${cmdPrefix}${PY_EXE} -m pytype --version

        ${cmdPrefix}${PY_EXE} -m pytype --config setup.cfg -j ${NUM_PARALLEL} --python-version="$(${PY_EXE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")" "$(pwd)"

        pytype_status=$?
        if [ ${pytype_status} -ne 0 ]
        then
            echo "${red}failed!${noColor}"
            exit ${pytype_status}
        else
            echo "${green}passed!${noColor}"
        fi
    fi
    set -e # enable exit on failure
fi


if [ $doMypyFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}mypy${noColor}"

    # ensure that the necessary packages for code format testing are installed
    ${cmdPrefix}${PY_EXE} -m mypy --version
    ${cmdPrefix}${PY_EXE} -m mypy "$(pwd)"

    mypy_status=$?
    if [ ${mypy_status} -ne 0 ]
    then
        : # mypy output already follows format
        exit ${mypy_status}
    else
        : # mypy output already follows format
    fi
    set -e # enable exit on failure
fi
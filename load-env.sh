#!/usr/bin/env bash

echo_and_do() {
    echo "$@"
    "$@"
}

if [[ $(hostname) =~ rzvernal ]]; then

    echo_and_do module load cmake/3.24.2
    echo_and_do module load rocm/5.2.0

else
    echo "UNRECOGNIZED PLATFORM $(hostname)"
fi
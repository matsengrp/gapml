#!/bin/bash
cd /efs/$1/gestaltamania/gestalt
python3 "${@:2}"

#!/bin/bash

# Get the data argument passed to the Bash script
data="$1"

# Construct the command to run the Python script with the data
command="python ../Gesture-Recognition/predict.py --gesture_loc \"$data\""
#command="python test.py"
#echo $data
# Execute the Python script with the data
eval "$command"

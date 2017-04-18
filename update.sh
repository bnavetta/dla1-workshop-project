#!/bin/bash

rsync -avz --exclude-from=.rsyncignore . training.us-east1-d.scratch-164900:project

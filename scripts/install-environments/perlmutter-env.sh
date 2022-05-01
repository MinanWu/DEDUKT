#!/bin/bash

CC=$(which cc)
CXX=$(which CC)

INSTALL_DIR=$SCRATCH/DEDUKT-$(git rev-parse --short HEAD)

#!/bin/bash

set -Eeuo pipefail

export RMQ_HOSTNAME=two.radical-project.org
export RMQ_PORT=33235
export RADICAL_PILOT_DBURL=mongodb://hyperrct:h1p3rrc7@two.radical-project.org:27017/hyperrct
export RADICAL_PILOT_PROFILE=True
export RADICAL_ENTK_PROFILE=True


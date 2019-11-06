#!/bin/bash

set -Eeuo pipefail

source env2/bin/activate
python2 src/entkdriver/__main__.py --uri $1

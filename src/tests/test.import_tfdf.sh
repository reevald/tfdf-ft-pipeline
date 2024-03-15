#!/bin/bash
set -euxo pipefail
python3 -c 'import tensorflow as tf; import tensorflow_decision_forests as tfdf; import tfx.v1 as tfx; print("tf version ==>", tf.__version__); print("tfdf version ==>", tfdf.__version__); print("tfx version ==>", tfx.__version__)'
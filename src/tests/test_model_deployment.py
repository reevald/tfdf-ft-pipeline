import logging
import os
import sys

import tensorflow as tf

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..")))

from src.modules.utils import F1ScoreBinaryBridge  # noqa: E402

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)


def test_latest_version_model_artifact() -> None:
    env_local_serving_model_dir = os.getenv("LOCAL_SERVING_MODEL_DIR")

    assert env_local_serving_model_dir, "Env LOCAL_SERVING_MODEL_DIR is None!"

    if not tf.io.gfile.exists(env_local_serving_model_dir):
        raise ValueError(f"Not found: {env_local_serving_model_dir}")

    list_models = os.listdir(env_local_serving_model_dir)
    if len(list_models) == 0:
        raise ValueError("Empty serving model directory, no one model version found!")

    # It should be only one version model available (latest version)
    # See: build/utils.py::get_latest_model_version
    latest_version_model = list_models[0]
    print("Latest version model that will be tested:", latest_version_model)

    model = tf.keras.models.load_model(
        filepath=os.path.join(env_local_serving_model_dir, latest_version_model),
        custom_objects={
            "F1ScoreBinaryBridge": F1ScoreBinaryBridge(
                # Based on baseline model performance
                average="macro",
                threshold=0.48,
            )
        },
    )
    logging.info("Model loaded successfully.")

    SERVING_DEFAULT_SIGNATURE_NAME = "serving_default"
    PREDICT_SIGNATURE_NAME = "predict"

    assert (
        SERVING_DEFAULT_SIGNATURE_NAME in model.signatures
    ), f"{SERVING_DEFAULT_SIGNATURE_NAME} not in model signature!"

    assert (
        PREDICT_SIGNATURE_NAME in model.signatures
    ), f"{PREDICT_SIGNATURE_NAME} not in model signature!"

    prediction_fn = model.signatures[SERVING_DEFAULT_SIGNATURE_NAME]

    test_prediction = prediction_fn(
        **{
            "act_exe_num": [[0]],
            "act_med_num": [[1]],
            "act_read_num": [[4]],
            "act_exe_nununique": [[0]],
            "act_med_nununique": [[1]],
            "act_read_nununique": [[1]],
            "task_exe_completed": [[0]],
            "task_med_completed": [[0]],
            "task_read_completed": [[0]],
            "streak_current": [[2]],
        }
    )

    assert test_prediction["probabilities"].shape == (
        1,
        2,
    ), f"Invalid output probabilities shape: {test_prediction['probabilities'].shape}"

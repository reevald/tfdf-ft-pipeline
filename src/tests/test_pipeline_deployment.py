import logging
import os
import sys

import tensorflow as tf

# Uncomment these lines if you want to try test locally with tests/.env file
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=".env")
from tfx import v1 as tfx
from tfx.v1.orchestration.metadata import sqlite_metadata_connection_config

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..")))

from src.tfx_pipelines import config, kubeflow_pipeline  # noqa: E402

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)


def test_e2e_pipeline() -> None:
    env_pipeline_name = os.getenv("PIPELINE_NAME")
    env_gcs_pipeline_root = os.getenv("GCS_PIPELINE_ROOT")
    env_gcs_data_root = os.getenv("GCS_DATA_ROOT")
    env_enable_anomaly_detection = os.getenv("ENABLE_ANOMALY_DETECTION")
    env_enable_tuning = os.getenv("ENABLE_TUNING")
    env_enable_cache = os.getenv("ENABLE_CACHE")
    env_enable_training_vertex = os.getenv("ENABLE_TRAINING_VERTEX")
    env_best_hparams = os.getenv("USER_PROVIDED_BEST_HYPERPARAMETER_PATH")
    env_enable_multi_processing = os.getenv("ENABLE_MULTI_PROCESSING")
    env_gcs_serving_model_dir = os.getenv("GCS_SERVING_MODEL_DIR")
    env_user_provided_schema_path = os.getenv("USER_PROVIDED_SCHEMA_PATH")
    env_module_transform_path = os.getenv("MODULE_TRANSFORM_PATH")
    env_module_tuner_path = os.getenv("MODULE_TUNER_PATH")
    env_module_trainer_path = os.getenv("MODULE_TRAINER_PATH")

    assert env_pipeline_name, "Env PIPELINE_NAME is None!"
    assert env_gcs_pipeline_root, "Env GCS_PIPELINE_ROOT is None!"
    assert env_gcs_data_root, "Env GCS_DATA_ROOT is None!"
    assert env_enable_anomaly_detection, "Env ENABLE_ANOMALY_DETECTION is None!"
    assert env_enable_tuning, "Env ENABLE_TUNING is None!"
    assert env_enable_cache, "Env ENABLE_CACHE is None!"
    assert env_enable_training_vertex, "Env ENABLE_TRAINING_VERTEX is None!"
    assert env_best_hparams, "Env USER_PROVIDED_BEST_HYPERPARAMETER_PATH is None!"
    assert env_enable_multi_processing, "Env ENABLE_MULTI_PROCESSING is None!"
    assert env_gcs_serving_model_dir, "Env GCS_SERVING_MODEL_DIR is None!"
    assert env_user_provided_schema_path, "Env USER_PROVIDED_SCHEMA_PATH is None!"
    assert env_module_transform_path, "Env MODULE_TRANSFORM_PATH is None!"
    assert env_module_tuner_path, "Env MODULE_TUNER_PATH is None!"
    assert env_module_trainer_path, "Env MODULE_TRAINER_PATH is None!"

    gcs_pipeline_root = config.GCS_PIPELINE_ROOT
    gcs_data_root = config.GCS_DATA_ROOT

    logging.info("Clean up previous test pipeline artifacts if exists.")
    if tf.io.gfile.exists(config.GCS_PIPELINE_ROOT):
        tf.io.gfile.rmtree(config.GCS_PIPELINE_ROOT)

    logging.info("Pipeline e2e test artifacts stored in: %s", gcs_pipeline_root)

    logging.info("Clean up previous test model serving directory if exists.")
    if tf.io.gfile.exists(config.GCS_SERVING_MODEL_DIR):
        tf.io.gfile.rmtree(config.GCS_SERVING_MODEL_DIR)

    # Since on the deployment the metadata will be handled by Vertex AI
    # then for e2e test, we can use metadata locally.
    metadata_pipeline_config = sqlite_metadata_connection_config(
        os.path.abspath("metadata.sqlite")
    )

    is_enable_anomaly_detection = config.ENABLE_ANOMALY_DETECTION == "YES"
    is_enable_tuning = config.ENABLE_TUNING == "YES"
    is_enable_cache = config.ENABLE_CACHE == "YES"
    is_enable_training_vertex = config.ENABLE_TRAINING_VERTEX == "YES"

    config_training_vertex = None

    best_hparams = config.USER_PROVIDED_BEST_HYPERPARAMETER_PATH

    beam_pipeline_direct_runner_args = []
    if config.ENABLE_MULTI_PROCESSING == "YES":
        beam_pipeline_direct_runner_args.append(
            "--direct_running_mode=multi_processing"
        )
        beam_pipeline_direct_runner_args.append(
            # 0 means auto-detect based on on the number of CPUs available
            # during execution time.
            "--direct_num_workers=0"
        )

    gcs_serving_model_dir = config.GCS_SERVING_MODEL_DIR

    runner = tfx.orchestration.LocalDagRunner()

    # In the test case, should be use only 1 epoch
    # and since Decision Forest using 1 epoch in whole training
    # then we don't need to setting up the epoch.
    # See: TFDF Caveats (https://github.com/tensorflow/decision-forests/blob/035529c1bc999d51caf25880d96e5813eef87736/tensorflow_decision_forests/keras/core.py#L326)
    # 1) The model trains for exactly one epoch. The core of the training
    # computation is done at the end of the first epoch. The console will
    # show training logs (including validations losses and feature statistics).

    # 2) During training, the entire dataset is loaded in memory (in an
    # efficient representation). In case of large datasets (>100M examples),
    # it is recommended to randomly downsample the examples.
    # We need to create little examples data for testing purpose.

    # Also we don't need to use training vertex AI and tuning component.

    pipeline = kubeflow_pipeline.create_pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=gcs_pipeline_root,
        data_root=gcs_data_root,
        module_transform_path=config.MODULE_TRANSFORM_PATH,
        module_tuner_path=config.MODULE_TUNER_PATH,
        module_trainer_path=config.MODULE_TRAINER_PATH,
        user_provided_schema_path=config.USER_PROVIDED_SCHEMA_PATH,
        user_provided_best_hyperparameter_path=best_hparams,
        enable_anomaly_detection=is_enable_anomaly_detection,
        enable_tuning=is_enable_tuning,
        enable_cache=is_enable_cache,
        enable_training_vertex=is_enable_training_vertex,
        config_training_vertex=config_training_vertex,
        metadata_connection_config=metadata_pipeline_config,
        serving_model_dir=gcs_serving_model_dir,
        beam_pipeline_args=beam_pipeline_direct_runner_args,
    )

    runner.run(pipeline=pipeline)

    logging.info("Model output: %s", gcs_serving_model_dir)

    assert tf.io.gfile.exists(gcs_serving_model_dir)

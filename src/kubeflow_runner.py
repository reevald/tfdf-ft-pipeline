import os
import sys
from typing import Any, Optional

from absl import logging
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner

# from tfx.v1.orchestration.metadata import sqlite_metadata_connection_config

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from src.tfx_pipelines import config, kubeflow_pipeline  # noqa: E402


def run_and_compile_pipeline(pipeline_definition_file: str) -> Optional[Any]:
    gcs_pipeline_root = config.GCS_PIPELINE_ROOT
    gcs_data_root = config.GCS_DATA_ROOT
    
    # We don't need to set metadata_connection_config which is used
    # to ML Metadata database. Because if using vertex, it will uses
    # a managed metadata services, we don't need to care of it.
    gcs_metadata_pipeline_config = None
    # metadata_pipeline_config = sqlite_metadata_connection_config(
    #     os.path.abspath("metadata.sqlite")
    # )

    gcs_serving_model_dir = config.GCS_SERVING_MODEL_DIR

    is_enable_anomaly_detection = config.ENABLE_ANOMALY_DETECTION == "YES"
    is_enable_tuning = config.ENABLE_TUNING == "YES"
    is_enable_cache = config.ENABLE_CACHE == "YES"
    is_enable_training_vertex = config.ENABLE_TRAINING_VERTEX == "YES"
    is_enable_ucaip = config.ENABLE_UCAIP == "YES"
    is_enable_gpu = config.ENABLE_GPU == "YES"

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
    
    vertex_args: dict = {
        "project": config.GOOGLE_CLOUD_PROJECT,
        "worker_pool_specs": [{
            "machine_spec": {
                "machine_type": "n1-standard-4",
                # In this case we don't use gpu.
                # "accelerator_type": "NVIDIA_TESLA_K80",
                # "accelerator_count": 1
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": config.TFX_IMAGE_URI
            }
        }]
    }

    config_training_vertex = {
        "IS_ENABLE_UCAIP": is_enable_ucaip,
        "UCAIP_REGION": config.UCAIP_REGION,
        "VERTEX_ARGS": vertex_args,
        "IS_USE_GPU": is_enable_gpu
    }

    managed_pipeline = kubeflow_pipeline.create_pipeline(
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
        metadata_connection_config=gcs_metadata_pipeline_config,
        serving_model_dir=gcs_serving_model_dir,
        beam_pipeline_args=beam_pipeline_direct_runner_args
    )

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            default_image=config.TFX_IMAGE_URI
        ),
        output_filename=pipeline_definition_file,
    )

    return runner.run(pipeline=managed_pipeline, write_out=True)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    pipeline_definition_file = f"{config.PIPELINE_NAME}.json"
    _ = run_and_compile_pipeline(pipeline_definition_file)

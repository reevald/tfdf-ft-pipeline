import os
import sys
from typing import Any, Optional

from absl import logging
from tfx import v1 as tfx

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from src.tfx_pipelines import config, local_pipeline  # noqa: E402


def run_pipeline() -> Optional[Any]:
    local_pipeline_root = os.path.abspath(config.PIPELINE_NAME)
    local_metadata_path = os.path.abspath(
        os.path.join(config.PIPELINE_NAME, "metadata.sqlite")
    )
    is_enable_anomaly_detection = config.ENABLE_ANOMALY_DETECTION == "YES"
    is_enable_tuning = config.ENABLE_TUNING == "YES"
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

    pipeline = local_pipeline.create_pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=local_pipeline_root,
        data_root=config.LOCAL_DATA_ROOT,
        enable_anomaly_detection=is_enable_anomaly_detection,
        user_provided_schema_path=config.USER_PROVIDED_SCHEMA_PATH,
        module_transform_path=config.MODULE_TRANSFORM_PATH,
        module_tuner_path=config.MODULE_TUNER_PATH,
        module_trainer_path=config.MODULE_TRAINER_PATH,
        user_provided_best_hyperparameter_path=best_hparams,
        enable_tuning=is_enable_tuning,
        serving_model_dir=config.LOCAL_SERVING_MODEL_DIR,
        metadata_path=local_metadata_path,
        beam_pipeline_args=beam_pipeline_direct_runner_args,
    )

    runner = tfx.orchestration.LocalDagRunner()
    return runner.run(pipeline=pipeline)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run_pipeline()

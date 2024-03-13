import os

MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "urge-classifier")
PIPELINE_NAME = os.getenv("PIPELINE_NAME", f"{MODEL_DISPLAY_NAME}-train-pipeline")

ENABLE_ANOMALY_DETECTION = os.getenv("ENABLE_ANOMALY_DETECTION", "YES")  # or "NO"
ENABLE_TUNING = os.getenv("ENABLE_TUNING", "NO")  # or "YES"
ENABLE_MULTI_PROCESSING = os.getenv("ENABLE_MULTI_PROCESSING", "NO")  # or "YES"

# =========================================================
# These must be consume from runner or interactive pipeline
# and executed from root project.

USER_PROVIDED_SCHEMA_PATH = os.getenv(
    "USER_PROVIDED_SCHEMA_PATH",
    os.path.abspath(os.path.join("src", "raw_schema", "schema.pbtxt")),
)
USER_PROVIDED_BEST_HYPERPARAMETER_PATH = os.getenv(
    "USER_PROVIDED_BEST_HYPERPARAMETER_PATH",
    os.path.abspath(os.path.join("src", "best_hyperparameters")),
)
MODULE_TRANSFORM_PATH = os.getenv(
    "MODULE_TRANSFORM_PATH",
    os.path.join("src", "modules", "transform_module.py"),
)
MODULE_TUNER_PATH = os.getenv(
    "MODULE_TUNER_PATH",
    os.path.abspath(os.path.join("src", "modules", "tuner_module.py")),
)
MODULE_TRAINER_PATH = os.getenv(
    "MODULE_TRAINER_PATH",
    os.path.abspath(os.path.join("src", "modules", "trainer_module.py")),
)

# =============== LOCAL ===============
LOCAL_LOCATION = os.getenv("LOCAL_LOCATION", ".")
LOCAL_DATA_ROOT = os.getenv(
    "LOCAL_DATA_ROOT",
    os.path.abspath(os.path.join(LOCAL_LOCATION, "sample_local_data")),
)
LOCAL_SERVING_MODEL_DIR = os.getenv(
    "LOCAL_SERVING_MODEL_DIR",
    os.path.abspath(
        os.path.join(LOCAL_LOCATION, "serving_saved_model", MODEL_DISPLAY_NAME)
    ),
)

# =============== GCS ===============
# To be support local, ensure
# GCS_
# GCS_DATA_ROOT =
# GCS_SERVING_MODEL_DIR =

import os

MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "urge-classifier")
PIPELINE_NAME = os.getenv("PIPELINE_NAME", f"{MODEL_DISPLAY_NAME}-train-pipeline")

ENABLE_ANOMALY_DETECTION = os.getenv("ENABLE_ANOMALY_DETECTION", "YES")  # "YES" or "NO"
ENABLE_TUNING = os.getenv("ENABLE_TUNING", "NO")  # "YES" or "NO"
ENABLE_MULTI_PROCESSING = os.getenv("ENABLE_MULTI_PROCESSING", "NO")  # "YES" or "NO"
ENABLE_TRAINING_VERTEX = os.getenv("ENABLE_TRAINING_VERTEX", "YES")  # "YES" or "NO"
ENABLE_UCAIP = os.getenv("ENABLE_UCAIP", "YES")  # "YES" or "NO"
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "NO")  # "YES" or "NO"
ENABLE_GPU = os.getenv("ENABLE_GPU", "NO")  # "YES" or "NO"

# =========================================================
# These must be consume from runner or interactive pipeline
# and executed from root project.

USER_PROVIDED_SCHEMA_PATH = os.getenv(
    "USER_PROVIDED_SCHEMA_PATH",
    os.path.join("src", "raw_schema", "schema.pbtxt"),
)
USER_PROVIDED_BEST_HYPERPARAMETER_PATH = os.getenv(
    "USER_PROVIDED_BEST_HYPERPARAMETER_PATH",
    os.path.join("src", "best_hyperparameters"),
)
MODULE_TRANSFORM_PATH = os.getenv(
    "MODULE_TRANSFORM_PATH",
    os.path.join("src", "modules", "transform_module.py"),
)
MODULE_TUNER_PATH = os.getenv(
    "MODULE_TUNER_PATH",
    os.path.join("src", "modules", "tuner_module.py"),
)
MODULE_TRAINER_PATH = os.getenv(
    "MODULE_TRAINER_PATH",
    os.path.join("src", "modules", "trainer_module.py"),
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
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "tfdf-mlops")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
DOCKER_REPO_NAME = os.getenv("DOCKER_REPO_NAME", "docker-repo")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "data-for-tfdf")

GCS_PIPELINE_ROOT = os.getenv(
    "GCS_PIPELINE_ROOT", f"gs://{GCS_BUCKET_NAME}/pipeline-root/{PIPELINE_NAME}"
)
GCS_DATA_ROOT = os.getenv(
    "GCS_DATA_ROOT", f"gs://{GCS_BUCKET_NAME}/data/{PIPELINE_NAME}"
)
GCS_SERVING_MODEL_DIR = os.getenv(
    "GCS_SERVING_MODEL_DIR", f"gs://{GCS_BUCKET_NAME}/serving-model/{PIPELINE_NAME}"
)
GCS_PIPELINES_STORE = os.getenv(
    "GCS_PIPELINES_STORE", f"gs://{GCS_BUCKET_NAME}/pipelines-store/{PIPELINE_NAME}"
)

TFX_IMAGE_URI = os.getenv(
    "TFX_IMAGE_URI",
    f"{GOOGLE_CLOUD_REGION}-docker.pkg.dev/{GOOGLE_CLOUD_PROJECT}/{DOCKER_REPO_NAME}/vertex:latest",
)

UCAIP_REGION = os.getenv("UCAIP_REGION", GOOGLE_CLOUD_REGION)

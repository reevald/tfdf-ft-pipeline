import argparse
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage  # type: ignore[attr-defined]

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True)

    parser.add_argument("--pipeline-name", type=str)

    parser.add_argument("--google-cloud-project", type=str)

    parser.add_argument("--google-cloud-region", type=str)

    parser.add_argument("--pipelines-store", type=str)

    parser.add_argument("--service-account", type=str)

    parser.add_argument("--gcs-serving-model-dir", type=str)

    parser.add_argument("--local-serving-model-dir", type=str)

    return parser.parse_args()


def compile_pipeline(pipeline_name: str) -> Optional[Any]:
    import src.kubeflow_runner as runner

    pipeline_definition_file = f"{pipeline_name}.json"
    result = runner.run_and_compile_pipeline(pipeline_definition_file)
    return result


def run_pipeline(
    pipeline_name: str,
    pipelines_store: str,
    gcp_project_id: str,
    gcp_region: str,
    service_account: str,
) -> None:
    storage_client = storage.Client()

    gcs_compiled_pipeline_file_location = (
        pipelines_store if pipelines_store.endswith("/") else pipelines_store + "/"
    )
    gcs_compiled_pipeline_file_location += pipeline_name + ".json"

    path_parts = gcs_compiled_pipeline_file_location.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    blob_name = "/".join(path_parts[1:])

    bucket = storage_client.bucket(bucket_name)
    blob = storage.Blob(bucket=bucket, name=blob_name)

    if not blob.exists(storage_client):
        raise ValueError(f"{blob_name} does not exist.")

    vertex_ai.init(project=gcp_project_id, location=gcp_region)

    job = vertex_ai.PipelineJob(
        display_name=pipeline_name, template_path=gcs_compiled_pipeline_file_location
    )

    job.submit(service_account=service_account, network=None)

    job.wait()
    print(f"Job finished with state: {job.state}")


def get_latest_model_version(
    gcs_serving_model_dir: str, local_serving_model_dir: str
) -> None:
    storage_client = storage.Client()

    path_parts = gcs_serving_model_dir.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    blob_name = "/".join(path_parts[1:])

    bucket = storage_client.bucket(bucket_name)
    blob_model_dir = storage.Blob(bucket=bucket, name=blob_name)

    if not blob_model_dir.exists(storage_client):
        raise ValueError(f"{blob_name} does not exist.")

    len_list_blob_name = len(path_parts[1:])
    list_blobs = list(bucket.list_blobs(prefix=blob_name + "/"))

    latest_num_version = 0
    last_check_str_version = ""
    for blob in list_blobs:
        str_version = str(blob.name).split("/")[len_list_blob_name]
        if len(str_version) > 0 and str_version != last_check_str_version:
            try:
                num_version = int(str_version)
                if num_version > latest_num_version:
                    latest_num_version = num_version
                last_check_str_version = str_version
            except ValueError as err:
                raise ValueError(f"Model version must be number. {err}") from err

    if latest_num_version == 0:
        raise ValueError("No model version in the given directory.")

    # Download latest num version of model into serving_saved_model
    target_download_dir = f"{blob_name}/{latest_num_version}/"
    local_download_dir = (
        local_serving_model_dir
        if local_serving_model_dir.endswith("/")
        else local_serving_model_dir + "/"
    )
    local_download_dir += f"{latest_num_version}/"
    Path(local_download_dir).mkdir(parents=True, exist_ok=True)

    blobs = bucket.list_blobs(prefix=target_download_dir)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.replace(target_download_dir, "").split("/")
        directory = local_download_dir + "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_download_dir + "/".join(file_split))
        logging.info(f"{blob.name} was successful downloaded.")


def main() -> None:
    args = get_args()
    if args.mode == "compile-pipeline":
        if not args.pipeline_name:
            raise ValueError("pipeline-name should be supplied.")
        result = compile_pipeline(args.pipeline_name)
    elif args.mode == "run-pipeline":
        if not args.google_cloud_project:
            raise ValueError("google-cloud-project should be supplied.")
        if not args.google_cloud_region:
            raise ValueError("google-cloud-region should be supplied.")
        if not args.pipeline_name:
            raise ValueError("pipeline-name should be supplied.")
        if not args.pipelines_store:
            raise ValueError("pipelines-store should be supplied.")
        if not args.service_account:
            raise ValueError("service-account should be supplied")

        result = run_pipeline(  # type: ignore[func-returns-value]
            pipeline_name=args.pipeline_name,
            pipelines_store=args.pipelines_store,
            gcp_project_id=args.google_cloud_project,
            gcp_region=args.google_cloud_region,
            service_account=args.service_account,
        )
    elif args.mode == "get-latest-model-version":
        if not args.gcs_serving_model_dir:
            raise ValueError("gcs-serving-model-dir should be supplied")
        if not args.local_serving_model_dir:
            raise ValueError("local-serving-model-dir should be supplied")

        result = get_latest_model_version(  # type: ignore[func-returns-value]
            gcs_serving_model_dir=args.gcs_serving_model_dir,
            local_serving_model_dir=args.local_serving_model_dir,
        )
    else:
        raise ValueError(f"Invalid mode {args.mode}")

    logging.info(result)


if __name__ == "__main__":
    main()

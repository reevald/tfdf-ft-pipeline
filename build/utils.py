import argparse
import logging
import os
import sys
from argparse import Namespace
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

    return parser.parse_args()


def compile_pipeline(pipeline_name: str) -> Optional[Any]:
    import src.kubeflow_runner as runner

    pipeline_definition_file = f"{pipeline_name}.json"
    result = runner.run_and_compile_pipeline(pipeline_definition_file)
    return result


def run_pipeline(
    pipeline_name: str, pipelines_store: str, gcp_project_id: str, gcp_region: str
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
        raise ValueError(f"{pipelines_store}/{pipeline_name} does not exist.")

    vertex_ai.init(project=gcp_project_id, location=gcp_region)

    job = vertex_ai.PipelineJob(
        display_name=pipeline_name,
        template_path=gcs_compiled_pipeline_file_location
    )

    job.submit()

    job.wait()
    print(f"Job finished with state: {job.state}")


def main() -> None:
    args = get_args()
    if args.mode == "compile-pipeline":
        if not args.pipeline_name:
            raise ValueError("pipeline-name should be supplied.")
        result = compile_pipeline(args.pipeline_name)
    elif args.mode == "run-pipeline":
        # pipeline_name: str, pipelines_store: str, gcp_project_id: str, gcp_region: str
        if not args.google_cloud_project:
            raise ValueError("google-cloud-project should be supplied.")
        if not args.google_cloud_region:
            raise ValueError("google-cloud-region should be supplied.")
        if not args.pipeline_name:
            raise ValueError("pipeline-name should be supplied.")
        if not args.pipelines_store:
            raise ValueError("pipelines-store should be supplied.")
        
        result = run_pipeline( # type: ignore[func-returns-value]
            pipeline_name=args.pipeline_name,
            pipelines_store=args.pipelines_store,
            gcp_project_id=args.google_cloud_project,
            gcp_region=args.google_cloud_region
        )
    else:
        raise ValueError(f"Invalid mode {args.mode}")

    logging.info(result)


if __name__ == "__main__":
    main()

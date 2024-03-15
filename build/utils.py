import os
import sys
import argparse
import logging

from typing import Any, Optional
from argparse import Namespace

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True
    )

    parser.add_argument(
        "--pipeline-name",
        type=str
    )

    return parser.parse_args()

def compile_pipeline(pipeline_name:str) -> Optional[Any]:
    import src.kubeflow_runner as runner
    pipeline_definition_file = f"{pipeline_name}.json"
    result = runner.run_and_compile_pipeline(pipeline_definition_file)
    return result

def main() -> None:
    args = get_args()
    if args.mode == "compile-pipeline":
        if not args.pipeline_name:
            raise ValueError("pipeline-name should be supplied.")
        result = compile_pipeline(args.pipeline_name)
    
    logging.info(result)

if __name__ == "__main__":
    main()

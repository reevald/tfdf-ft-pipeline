from typing import Any, List, Optional, Union

import tensorflow_model_analysis as tfma
import tfx.v1 as tfx
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.types import Channel, standard_artifacts
from tfx.v1.extensions.google_cloud_ai_platform import Trainer as VertexTrainer


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_transform_path: str,
    module_tuner_path: str,
    module_trainer_path: str,
    user_provided_schema_path: str,
    user_provided_best_hyperparameter_path: str,
    enable_anomaly_detection: bool,
    enable_tuning: bool,
    enable_cache: bool,
    enable_training_vertex: bool,
    config_training_vertex: Optional[dict],
    serving_model_dir: str,
    metadata_connection_config: Optional[Any],
    beam_pipeline_args: Optional[List[Union[str, Any]]],
) -> tfx.dsl.Pipeline:
    # TODO: separate between train and test data to prevent data leak,
    # the test data only used in evaluator.
    example_gen_input_config = tfx.proto.Input(
        splits=[
            tfx.proto.Input.Split(
                name="train",
                pattern="{YYYY}-{MM}-{DD}/ver-{VERSION}/train/synth_train.csv",
            ),
            tfx.proto.Input.Split(
                name="val",
                pattern="{YYYY}-{MM}-{DD}/ver-{VERSION}/val/synth_val.csv",
            ),
        ]
    )

    example_gen = tfx.components.CsvExampleGen(
        input_base=data_root, input_config=example_gen_input_config
    ).with_id("ExampleGen")

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    ).with_id("StatisticsGen")

    schema_gen: Union[tfx.components.SchemaGen, tfx.components.ImportSchemaGen]
    if enable_anomaly_detection:
        # Import user-provided schema.
        schema_gen = tfx.components.ImportSchemaGen(
            schema_file=user_provided_schema_path
        ).with_id("SchemaImporter")
        # Performs anomaly detection based on statistics and data schema.
        example_validator = tfx.components.ExampleValidator(
            statistics=statistics_gen.outputs["statistics"],
            schema=schema_gen.outputs["schema"],
        ).with_id("ExampleValidator")
    else:
        # Generates schema based on statistics files.
        schema_gen = tfx.components.SchemaGen(
            statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
        ).with_id("SchemaGen")

    # Performs transformations and feature engineering in training and serving.
    # TODO: improve with Apache Beam (Google Dataflow)
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_transform_path,
        custom_config={
            "path_schema": user_provided_schema_path,
        },
    ).with_id("DataTransformer")

    # Tunes the hyperparameters for model training based on user-provided Python
    # function (check on tuner module). Note that once the hyperparameters are
    # tuned, we can drop the Tuner component from pipeline (for efficiency)
    # and feed Trainer with tuned hyperparameters.
    if enable_tuning:
        tuner = tfx.components.Tuner(
            examples=transform.outputs["transformed_examples"],
            schema=schema_gen.outputs["schema"],
            transform_graph=transform.outputs["transform_graph"],
            train_args=tfx.proto.TrainArgs(splits=["train"]),
            eval_args=tfx.proto.EvalArgs(splits=["val"]),
            module_file=module_tuner_path,
            custom_config={
                # Based on benchmark of baseline models
                "model_type": "GBTM",
                "best_threshold": 0.48,
                "input_fn_version": "V1",
                "tuner_type": "GST",
                "max_trials": 30,
                "batch_size": 128,
                "label_key": "streak_status",
            },
        ).with_id("ModelTuner")

    else:
        hparams_importer = tfx.dsl.Importer(
            source_uri=user_provided_best_hyperparameter_path,
            artifact_type=tfx.types.standard_artifacts.HyperParameters,
        ).with_id("HyperparametersImporter")

    # Get the latest model to warmstart
    warmstart_model_resolver = Resolver(
        strategy_class=latest_artifacts_resolver.LatestArtifactsResolver,
        latest_model=Channel(type=standard_artifacts.Model),
    ).with_id("WarmstartModelResolver")

    trainer = tfx.components.Trainer(
        examples=transform.outputs["transformed_examples"],
        schema=transform.outputs["post_transform_schema"],
        base_model=warmstart_model_resolver.outputs["latest_model"],
        transform_graph=transform.outputs["transform_graph"],
        # If Tuner is in the pipeline, Trainer can take Tuner's output
        # best_hyperparameters artifact as input and utilize it in the user module code.
        # If there isn't Tuner in the pipeline, either use Importer to import
        # a previous Tuner's output to feed to Trainer, or directly use the tuned
        # hyperparameters in user module code and set hyperparameters to None here.
        hyperparameters=tuner.outputs["best_hyperparameters"]
        if enable_tuning
        else hparams_importer.outputs["result"],
        train_args=tfx.proto.TrainArgs(splits=["train"]),
        eval_args=tfx.proto.EvalArgs(splits=["val"]),
        module_file=module_trainer_path,
        custom_config={
            # Based on benchmark of baseline models
            "model_type": "GBTM",
            "input_fn_version": "V1",
            "batch_size": 128,
            "label_key": "streak_status",
        },
    ).with_id("ModelTrainer")

    if enable_training_vertex:
        if config_training_vertex is None:
            raise ValueError("Config should be provided if enable training vertex")
        
        try:
            enable_ucaip = config_training_vertex["IS_ENABLE_UCAIP"]
            ucaip_region = config_training_vertex["UCAIP_REGION"]
            vertex_args = config_training_vertex["VERTEX_ARGS"]
            use_gpu = config_training_vertex["IS_USE_GPU"]

        except KeyError as err:
            raise KeyError(f"Missing key in config training vertex. {err}") from err

        trainer = VertexTrainer(
            examples=transform.outputs["transformed_examples"],
            schema=transform.outputs["post_transform_schema"],
            base_model=warmstart_model_resolver.outputs["latest_model"],
            transform_graph=transform.outputs["transform_graph"],
            # If Tuner is in the pipeline, Trainer can take Tuner's output
            # best_hyperparameters artifact as input and utilize in the module code.
            # If there isn't Tuner in the pipeline, either use Importer to import
            # a previous Tuner's output to feed to Trainer, or directly use the tuned
            # hyperparameters in user module code and set hyperparameters to None here.
            hyperparameters=tuner.outputs["best_hyperparameters"]
            if enable_tuning
            else hparams_importer.outputs["result"],
            train_args=tfx.proto.TrainArgs(splits=["train"]),
            eval_args=tfx.proto.EvalArgs(splits=["val"]),
            module_file=module_trainer_path,
            custom_config={
                # Based on benchmark of baseline models
                "model_type": "GBTM",
                "input_fn_version": "V1",
                "batch_size": 128,
                "label_key": "streak_status",
                # Vertex config
                tfx.extensions.google_cloud_ai_platform.ENABLE_UCAIP_KEY: enable_ucaip,
                tfx.extensions.google_cloud_ai_platform.UCAIP_REGION_KEY: ucaip_region,
                tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_args,
                "use_gpu": use_gpu,
            },
        ).with_id("ModelVertexTrainer")

    # Get the latest blessed model (baseline) for model validation
    baseline_model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing),
    ).with_id("BaselineModelResolver")

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="eval_for_tfma",
                label_key="streak_status",
                preprocessing_function_names=["transform_features"],
            )
        ],
        slicing_specs=[
            # An empty slice spec means the overall slice, i.e. the whole dataset.
            tfma.SlicingSpec()
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        module="src.modules.utils",
                        class_name="F1ScoreBinaryBridge",
                        config="""
                            "average": "micro",
                            "name": "f1_score_micro",
                            "threshold": 0.48
                        """,
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                # For synthetic data still good even use micro since the
                                # data already balanced
                                lower_bound={
                                    # Based on baseline model performance
                                    "value": 0.737500
                                }
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": 0.01},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(
                        class_name="AUC",
                        config='"curve": "ROC"',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={
                                    # Based on baseline model performance
                                    "value": 0.779446
                                }
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": 0.01},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        # We will use default threshold (0.5) since the lower bound use
                        # it as well See ModelZooDF on modules/zoo.py module
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={
                                    # Based on baseline model performance
                                    "value": 0.728241
                                }
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": 0.01},
                            ),
                        ),
                    ),
                    # Necessary metrics for binary classification case
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="FalsePositives"),
                    tfma.MetricConfig(class_name="TruePositives"),
                    tfma.MetricConfig(class_name="FalseNegatives"),
                    tfma.MetricConfig(class_name="TrueNegatives"),
                ],
            )
        ],
    )

    # Model evaluation
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs["examples"],
        example_splits=["train", "val"],
        model=trainer.outputs["model"],
        baseline_model=baseline_model_resolver.outputs["model"],
        eval_config=eval_config,
    ).with_id("ModelEvaluator")

    pusher = tfx.components.Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    component_list: List[tfx.types.BaseNode] = [
        example_gen,
        statistics_gen,
        schema_gen,
        transform,
        warmstart_model_resolver,
        trainer,
        baseline_model_resolver,
        evaluator,
        pusher
    ]

    if enable_anomaly_detection:
        component_list.append(example_validator)
    if enable_tuning:
        component_list.append(tuner)
    else:
        component_list.append(hparams_importer)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        data_root=data_root,
        components=component_list,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )

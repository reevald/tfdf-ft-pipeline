"""
Trainer Module: to be consumed by Trainer component and handle train the model.
"""

from typing import Any, Dict, List, Tuple

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
import tfx.v1 as tfx

from src.modules.utils import (
    cm_builder,
    gbtm_builder,
    input_fn_v1,
    input_fn_v2,
    input_fn_v3,
    rfm_builder,
)


def get_serve_tf_examples_fn(
    model: tf.keras.Model, tf_transform_output: tft.TFTransformOutput, label_key:str
) -> Dict[str, Any]:
    """Returns the output to be used in the serving signature.

    Args:
        model: tf.keras.Model (model to be used during training)
        tf_transform_output: TFTransformOutput (output from Transform component)

    Returns:
        dict of serving signature
    """
    # We need to track the layers in the model in order to save it.
    # model.tft_layer workflow: raw_feature => transformed_feature
    model.tft_layer = tf_transform_output.transform_features_layer()
    raw_label_key = label_key

    # Input data as serialized to boost performance when inference the models
    # See: https://medium.com/@avidaneran/tensorflow-serving-rest-vs-grpc-e8cef9d4ff62
    @tf.function
    def serialized_handler(serialized_tf_examples: tf.Tensor) -> tf.Tensor:
        # Returns the output to be used in the serving signature
        raw_feature_spec = tf_transform_output.raw_feature_spec()

        # Remove label feature since this will not be present at serving
        raw_feature_spec.pop(raw_label_key)
        parsed_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    @tf.function
    def transform_features_fn(serialized_tf_example: tf.Tensor) -> tf.Tensor:
        # All labels are included (+ target label)
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)
        return transformed_features

    @tf.function
    def signature_serving_default(
        **kwargs_tensorspec: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        # Now without serialize to make more independent when generate input model
        transformed_features = model.tft_layer(kwargs_tensorspec)
        single_logits = model(transformed_features)
        # In this case we have two class (binary case)
        # Ex (2 input): [[0.75], [0.2]] => [[0, 0.75], [0, 0,2]]
        binary_logits = tf.concat(
            [tf.zeros_like(single_logits), single_logits], axis=-1
        )

        return {
            # Convert logits to softer predict probabilities by softmax
            # Ex (each input): [[0, 0.75]] => [[0.32, 0.68]]
            # Formula: probability(x) = exp(x)/sum(exp(i)) for i = {0.32, 0.68}
            "probabilities": tf.keras.layers.Softmax()(binary_logits)
        }

    # This predict signature will be main consumed by api (REST / gRPC) later
    @tf.function
    def signature_predict(
        serialized_examples: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        single_logits = serialized_handler(serialized_examples)
        binary_logits = tf.concat(
            [tf.zeros_like(single_logits), single_logits], axis=-1
        )
        # Enrich the output
        return {
            "logits": single_logits,
            # logistic from single_logits (x) equals to probability(x) above [[0.68]]
            "logistic": tf.keras.layers.Activation("sigmoid")(single_logits),
            "probabilities": tf.keras.layers.Softmax()(binary_logits),
        }

    return {
        # Optimize with get_concrete_function (reduce tracing stage)
        # The order of the signature functions matter (TFMA => Inference)
        "eval_for_tfma": serialized_handler.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
        "transform_features": transform_features_fn.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
        "serving_default": signature_serving_default.get_concrete_function(
            **{
                key: tf.TensorSpec(shape=[None, 1], dtype=tf.int64)
                for key in tf_transform_output.raw_feature_spec()
                if key != raw_label_key
            }
        ),
        "predict": signature_predict.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="inputs")
        ),
    }


def get_split_dataset( #pylint: disable=too-many-arguments
    train_files: List[str],
    eval_files: List[str],
    data_accessor: tfx.components.DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int,
    label_key: str,
    input_fn_version: str
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Get train and val dataset from the given train and eval files.

    Args:
        train_files: list of paths or patterns of input train tfrecord files.
        eval_files: list of paths or pattern of input eval tfrecord files.
        data_accessor: data accessor that will be used inside dataset factory.
        tf_transform_output: artifact output from transform component.
        batch_size: batch size of the dataset.
        label_key: label or feature target of the dataset.

    Returns:
        pair of train and val dataset that accorded with transform component output.
    """
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    if input_fn_version == "V1":
        train_dataset = input_fn_v1(
            file_pattern=train_files,
            tf_transform_output=tf_transform_output, # pylint: disable=duplicate-code
            batch_size=batch_size,
            label_key=label_key,
        )
        val_dataset = input_fn_v1(
            file_pattern=eval_files,
            tf_transform_output=tf_transform_output, # pylint: disable=duplicate-code
            batch_size=batch_size,
            label_key=label_key,
        )

    if input_fn_version == "V2":
        train_dataset = input_fn_v2(
            file_pattern=train_files,
            data_accessor=data_accessor,
            tf_transform_output=tf_transform_output, # pylint: disable=duplicate-code
            batch_size=batch_size,
            label_key=label_key,
        )
        val_dataset = input_fn_v2(
            file_pattern=eval_files,
            data_accessor=data_accessor,
            tf_transform_output=tf_transform_output, # pylint: disable=duplicate-code
            batch_size=batch_size,
            label_key=label_key,
        )

    if input_fn_version == "V3":
        train_dataset = input_fn_v3(
            file_pattern=train_files,
            tf_transform_output=tf_transform_output, # pylint: disable=duplicate-code
            batch_size=batch_size,
            label_key=label_key,
        )
        val_dataset = input_fn_v3(
            file_pattern=eval_files,
            tf_transform_output=tf_transform_output, # pylint: disable=duplicate-code
            batch_size=batch_size,
            label_key=label_key,
        )

    return train_dataset, val_dataset


def run_fn(fn_args: tfx.components.FnArgs) -> None:
    """Main function that will be executed by Trainer component.

    Args:
        fn_args: dictionary of configurations that filled by Trainer component.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    batch_size = fn_args.custom_config.get("batch_size")
    label_key = fn_args.custom_config.get("label_key")
    model_type = fn_args.custom_config.get("model_type")
    input_fn_version = fn_args.custom_config.get("input_fn_version")

    train_dataset, val_dataset = get_split_dataset(
        train_files=fn_args.train_files,
        eval_files=fn_args.eval_files,
        data_accessor=fn_args.data_accessor,
        tf_transform_output=tf_transform_output,
        batch_size=batch_size, # type: ignore
        label_key=label_key, # type: ignore
        input_fn_version=input_fn_version # type: ignore
    )

    hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
    # pylint: disable=duplicate-code
    if model_type == "RFM":
        model_builder = rfm_builder

    if model_type == "GBTM":
        model_builder = gbtm_builder

    if model_type == "CM":
        model_builder = cm_builder

    model = model_builder(hparams=hparams)
    model.fit(train_dataset, validation_data=val_dataset, verbose=1)

    # Export the tensorboard logs
    model.make_inspector().export_to_tensorboard(fn_args.model_run_dir)

    model.save(
        filepath=fn_args.serving_model_dir,
        overwrite=True,
        save_format="tf",
        signatures=get_serve_tf_examples_fn(
            model=model,
            tf_transform_output=tf_transform_output,
            label_key=label_key # type: ignore
        ),
    )

"""
Zoo Module: to compare input_fn and baseline model (TFDF)
"""

from timeit import default_timer as timer
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_decision_forests as tfdf
import tensorflow_transform as tft
from tfx.components.util import tfxio_utils
from tfx.types.artifact import Artifact
from tfx.v1.components import DataAccessor

from src.modules.utils import input_fn_v1, input_fn_v2, input_fn_v3


class InputFnZoo:
    """The main purpose of this module to help determine the input_fn version with
    optimum time execution when handling the data
    """

    def __init__(
        self,
        tf_transform_output: tft.TFTransformOutput,
        batch_size: int,
        transform_examples_artifacts: List[Artifact],
    ) -> None:
        self.best_input_fn_version: pd.Series
        self.tf_transform_output = tf_transform_output
        self.batch_size = batch_size
        self.transform_artifacts = transform_examples_artifacts
        self.telemetry_desc = ["transform"]
        self.data_accessor = DataAccessor(
            tf_dataset_factory=tfxio_utils.get_tf_dataset_factory_from_artifact(
                self.transform_artifacts, self.telemetry_desc
            ),
            record_batch_factory=tfxio_utils.get_record_batch_factory_from_artifact(
                self.transform_artifacts, self.telemetry_desc
            ),
            data_view_decode_fn=tfxio_utils.get_data_view_decode_fn_from_artifact(
                self.transform_artifacts, self.telemetry_desc
            ),
        )

    def get_dataset_with_best_input_fn(
        self, file_pattern: List[str], label_key: str
    ) -> tf.data.Dataset:
        """Get tf.data.Dataset with best input_fn from result of compared input_fn
        version. Make sure compare first before run this function.

        Args:
            file_pattern: list of paths or patterns of input tfrecord files
            label_key: feature key of dataset for label (target feature)

        Returns:
            dataset with tf.data.Dataset format
        """
        result_data: tf.data.Dataset
        if self.best_input_fn_version is None:
            raise ValueError("You must run compare() first to get best input fn")
        if self.best_input_fn_version["Mean"] == "V1":
            result_data = input_fn_v1(
                file_pattern=file_pattern,
                tf_transform_output=self.tf_transform_output,
                batch_size=self.batch_size,
                label_key=label_key,
            )
        if self.best_input_fn_version["Mean"] == "V2":
            result_data = input_fn_v2(
                file_pattern=file_pattern,
                tf_transform_output=self.tf_transform_output,
                batch_size=self.batch_size,
                data_accessor=self.data_accessor,
                label_key=label_key,
            )
        if self.best_input_fn_version["Mean"] == "V3":
            result_data = input_fn_v3(
                file_pattern=file_pattern,
                tf_transform_output=self.tf_transform_output,
                batch_size=self.batch_size,
                label_key=label_key,
            )
        return result_data

    def compare(
        self, file_pattern: List[str], label_key: str, num_run: int = 5
    ) -> pd.DataFrame:
        """Compare several input_fn version to determine which one that
        has minimum time execution to handle the data.

        Args:
            file_pattern: list of paths or patterns of input tfrecord files
            label_key: feature key of dataset for label (target feature)
            num_run: number of test running

        Returns:
            summary of time execution comparison (pandas dataframe format)
        """
        dict_comb: Dict[str, Any] = {
            "Input Fn Version": ["V1", "V2", "V3"],
        }
        for num in range(1, num_run + 1):
            label = f"Time Exec (ms) Run #{num}"
            dict_comb[label] = []
            # Version 1
            start = timer()
            input_fn_v1(
                file_pattern=file_pattern,
                tf_transform_output=self.tf_transform_output,
                batch_size=self.batch_size,
                label_key=label_key,
            )
            end = timer()
            dict_comb[label].append((end - start) * (10**3))
            # Version 2
            start = timer()
            input_fn_v2(
                file_pattern=file_pattern,
                tf_transform_output=self.tf_transform_output,
                batch_size=self.batch_size,
                data_accessor=self.data_accessor,
                label_key=label_key,
            )
            end = timer()
            dict_comb[label].append((end - start) * (10**3))
            # Version 3
            start = timer()
            input_fn_v3(
                file_pattern=file_pattern,
                tf_transform_output=self.tf_transform_output,
                batch_size=self.batch_size,
                label_key=label_key,
            )
            end = timer()
            dict_comb[label].append((end - start) * (10**3))

        df_result = pd.DataFrame.from_dict(dict_comb)
        df_result = df_result.set_index("Input Fn Version")
        df_result["Mean"] = df_result.mean(axis=1)
        self.best_input_fn_version = df_result[["Mean"]].idxmin()
        return df_result


class ModelDFZoo:
    """User entry point for the Model Zoo Decision Forest API.
    The main purpose of this module to help determine the initial model [either
    Random Forest, Gradient Boosted Trees or Cart Model] before we go further
    like training and tuning process.

    See: https://github.com/tensorflow/decision-forests
    """

    def __init__(self) -> None:
        self.train_ds: tf.data.Dataset
        self.valid_ds: tf.data.Dataset
        self.mean_metrics: pd.Series

    def tuning_threshold(  # pylint: disable=too-many-locals
        self,
        model: tf.keras.Model,
        min_range: float = 0.4,
        max_range: float = 0.8,
        step: float = 0.01,
    ) -> Tuple[float, float, float]:
        """To help determine best threshold for imbalanced of the distribution class

        Args:
            model: class model of keras Decision Forest
            min: minimum number of threshold (>= 0.0)
            max: maximum number of threshold (> min and <= 1.0)
            step: step to move threshold between min to max

        Return:
            outputs: tuple (f1-score and best threshold)
        """
        opt_threshold = 0.0
        opt_f1_score_macro = 0.0
        opt_f1_score_micro = 0.0
        for obs_threshold in np.arange(min_range, max_range, step):
            predict = model.predict(x=self.valid_ds, verbose=0)
            # Example transform [[0.76]] to [[0, 1]] if 0.76 >= threshold
            # another example [[0.2]] to [[1, 0]] if 0.2 < threshold
            y_pred = tf.one_hot(
                (predict.reshape(-1) >= obs_threshold).astype("int"), depth=2
            )
            # Example transform [[0],[1]] => [[1, 0], [0, 1]]
            list_ohe = []
            for batch_data in list(self.valid_ds.as_numpy_iterator()):
                list_ohe.append(tf.one_hot(batch_data[1].astype("int"), depth=2))
            y_true = np.concatenate(tuple(list_ohe), axis=0)
            # We use average macro for binary classes
            metric_f1_macro = tfa.metrics.F1Score(
                num_classes=2, average="macro", threshold=obs_threshold
            )
            metric_f1_macro.update_state(y_true=y_true, y_pred=y_pred)
            f1_score_macro = metric_f1_macro.result().numpy()
            if f1_score_macro > opt_f1_score_macro:
                metric_f1_micro = tfa.metrics.F1Score(
                    num_classes=2, average="micro", threshold=obs_threshold
                )
                metric_f1_micro.update_state(y_true=y_true, y_pred=y_pred)
                f1_score_micro = metric_f1_micro.result().numpy()

                opt_f1_score_micro = f1_score_micro
                opt_f1_score_macro = f1_score_macro
                opt_threshold = obs_threshold

        return (opt_f1_score_micro, opt_f1_score_macro, opt_threshold)

    def compare(
        self, train_ds: tf.data.Dataset, valid_ds: tf.data.Dataset
    ) -> pd.DataFrame:
        """Compare several model from TensorFlow Decision Forest such as
        Random Forest, Gradient Boosted Trees, and Cart Model. Each model
        will be evaluated with accuracy and f1-score metrics (with threshold-tuning).

        Args:
            train_ds: train dataset (tf.data format / prefetchDataset)
            val_ds: valid dataset (tf.data format / prefetchDataset)

        Return:
            outputs: summary of models comparison (pandas dataframe format)
        """
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        dict_comb: Dict[str, Any] = {
            "Model Name": [],
            "Accuracy": [],
            "Val Accuracy": [],
            "AUC": [],
            "Val AUC": [],
            "Val F1 Score Micro": [],
            "Val F1 Score Macro": [],
            "Best Threshold": [],
        }
        for model in tfdf.keras.get_all_models()[:3]:
            obs_model = model(verbose=0)
            obs_model.compile(
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.AUC(curve="ROC", name="auc"),
                ],
            )
            # Train the model
            obs_model.fit(self.train_ds)
            # Evaluation
            result_eval = obs_model.evaluate(
                x=self.train_ds, return_dict=True, verbose=0
            )
            result_val_eval = obs_model.evaluate(
                x=self.valid_ds, return_dict=True, verbose=0
            )
            (
                val_f1_score_micro,
                val_f1_score_macro,
                best_threshold,
            ) = self.tuning_threshold(obs_model)
            dict_comb["Model Name"].append(obs_model.name)
            dict_comb["Accuracy"].append(result_eval["binary_accuracy"])
            dict_comb["Val Accuracy"].append(result_val_eval["binary_accuracy"])
            dict_comb["AUC"].append(result_eval["auc"])
            dict_comb["Val AUC"].append(result_val_eval["auc"])
            dict_comb["Val F1 Score Micro"].append(val_f1_score_micro)
            dict_comb["Val F1 Score Macro"].append(val_f1_score_macro)
            dict_comb["Best Threshold"].append(best_threshold)

        df_result = pd.DataFrame.from_dict(dict_comb)
        df_result = df_result.set_index("Model Name")
        self.mean_metrics = df_result.drop("Best Threshold", axis=1).mean(axis=0)
        return df_result

import glob
import json
import sqlite3
import time
from typing import Union
from pathlib import Path
from loguru import logger

from metrics.exact_match import SpiderExactMatchMetric
from metrics.execution_accuracy import SpiderExecutionAccuracyMetric
from metrics.partial_match import PartialMatchMetric


def _log_errors(errors: list[dict], save_dir: str, errors_type: str) -> None:

    save_path = f"{save_dir}/{errors_type}.json" if save_dir is not None else (f"storage/{errors_type}/"
                                                                                 f"{time.time()}.json")
    Path("/".join(save_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write(json.dumps(errors))

    logger.info(f"The {errors_type} have been successfully saved in {save_path}.")


def calculate_metrics(predictions: list[str], references: list[dict[str, str]], calc_exec: bool = True,
                      save_errors: bool = True, errors_dir: str = None) -> list[dict[str, Union[float, bool]]]:
    metrics = {}
    exact_match_metric = SpiderExactMatchMetric()

    exact_match_results = exact_match_metric.compute(predictions=predictions, references=references,
                                                     not_aggregated=True, return_errors=True)

    metrics = {"EM": [result["exact"] for result in exact_match_results["results"]]}

    if save_errors and len(exact_match_results["errors"]):
        _log_errors(errors=exact_match_results["errors"], save_dir=errors_dir, errors_type="exact_match_errors")

    if calc_exec:
        exec_acc_metric = SpiderExecutionAccuracyMetric()

        exec_results = exec_acc_metric.compute(predictions=predictions, references=references,
                                               not_aggregated=True, return_errors=True)
        metrics["EX"] = [result["exec"] for result in exec_results["results"]]

        if save_errors and len(exec_results["errors"]):
            _log_errors(errors=exec_results["errors"], save_dir=errors_dir, errors_type="execution_accuracy_errors")
    else:
        # Set all values of execution accuracy to 0 (because of evaluation)
        metrics["EX"] = [1 for _ in range(len(predictions))]

    partial_match_metric = PartialMatchMetric()

    partial_matches = partial_match_metric.compute(predictions=predictions,
                                                   references=references,
                                                   not_aggregated=True,
                                                   extensive_report=False)

    for value in partial_match_metric.reported_values(extensive=False):
        metrics[value] = []

    for result in partial_matches:
        for value in partial_match_metric.reported_values(extensive=False):
            metrics[value].append(result[value])

    metrics = [dict(zip(metrics.keys(), values)) for values in zip(*metrics.values())]

    return metrics


def add_metrics_in_file_with_predictions(file_path: str, mappings: dict[str, str], calc_exec: bool = True):
    print(f"Add metrics in {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = data["per_item"] if "per_item" in data else data

    # Get references and predictions
    predictions = []
    references = []
    for datapoint in dataset:
        predictions.append(datapoint[mappings["prediction"] if "prediction" in mappings else "predicted"])

        if "db_path" in mappings:
            db_path = mappings["db_path"].format(db_path=datapoint["db_path"],
                                                 db_id=datapoint["db_id"]) if "db_path" in datapoint \
                else mappings["db_path"].format(db_id=datapoint["db_id"])
        else:
            db_path = datapoint["db_path"]

        references.append({
            "sql_query": datapoint[mappings["sql_query"]] if "sql_query" in mappings else datapoint["sql_query"],
            "db_id": datapoint["db_id"],
            "db_path": db_path
        })

    # Calculate metrics
    metrics = calculate_metrics(predictions=predictions, references=references, calc_exec=calc_exec, save_errors=True,
                                errors_dir=f"storage/metrics_errors/{file_path.split('/')[-1][:-5]}")

    for datapoint, datapoint_metrics in zip(dataset, metrics):
        datapoint["metrics"] = datapoint_metrics

    with open(file_path, "w") as f:
        f.write(json.dumps(data))


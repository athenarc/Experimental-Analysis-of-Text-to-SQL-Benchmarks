import json
import sqlite3
from typing import Union

from metrics.exact_match import SpiderExactMatchMetric
from metrics.execution_accuracy import SpiderExecutionAccuracyMetric
from metrics.partial_match import PartialMatchMetric


def calculate_metrics(predictions: list[str], references: list[dict[str, str]], calc_exec: bool = True) -> \
        list[dict[str, Union[float, bool]]]:
    exact_match_metric = SpiderExactMatchMetric()
    metrics = {"EM": [result["exact"] for result in exact_match_metric.compute(predictions=predictions,
                                                                               references=references,
                                                                               not_aggregated=True)]}

    if calc_exec:
        exec_acc_metric = SpiderExecutionAccuracyMetric()
        try:
            metrics["EX"] = [result["exec"] for result in exec_acc_metric.compute(predictions=predictions,
                                                                                  references=references,
                                                                                  not_aggregated=True)]
        except sqlite3.OperationalError:
            metrics["EX"] = [0 for _ in range(len(predictions))]
    else:
        metrics["EX"] = [None for _ in range(len(predictions))]

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
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = data["per_item"] if "per_item" in data else data

    # Get references and predictions
    predictions = []
    references = []
    for datapoint in dataset:
        predictions.append(datapoint[mappings["prediction"]])

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
    metrics = calculate_metrics(predictions=predictions, references=references, calc_exec=calc_exec)

    for datapoint, datapoint_metrics in zip(dataset, metrics):
        datapoint["metrics"] = datapoint_metrics

    with open(file_path, "w") as f:
        f.write(json.dumps(data))


if __name__ == "__main__":

    # Evaluation results on models from others
    evaluation_files = [
        {
            "path": "storage/evaluation_files/spider/models_from_others/t5_base_eval.json",
            "mappings": {"sql_query": "gold", "db_path": "third_party/spider/database/{db_id}/{db_id}.sqlite",
                         "prediction": "predicted"}
        },
        {
            "path": "storage/evaluation_files/spider/models_from_others/t5_large_eval.json",
            "mappings": {"sql_query": "gold", "db_path": "third_party/spider/database/{db_id}/{db_id}.sqlite",
                         "prediction": "predicted"}
        },
        {
            "path": "storage/evaluation_files/spider/models_from_others/t5_large_lm100k_eval.json",
            "mappings": {"sql_query": "gold", "db_path": "third_party/spider/database/{db_id}/{db_id}.sqlite",
                         "prediction": "predicted"}
        },
        {
            "path": "storage/evaluation_files/spider/models_from_others/t5_3b_eval.json",
            "mappings": {"sql_query": "gold", "db_path": "third_party/spider/database/{db_id}/{db_id}.sqlite",
                         "prediction": "predicted"}
        },
        {
            "path": "storage/evaluation_files/spider/models_from_others/DIN-SQL.json",
            "mappings": {"sql_query": "gold", "db_path": "third_party/spider/database/{db_id}/{db_id}.sqlite",
                         "prediction": "predicted"}
        },
        {
            "path": "storage/evaluation_files/spider/models_from_others/bart_run_1_true_1-step41000.eval",
            "mappings": {"sql_query": "gold", "db_path": "third_party/spider/database/{db_id}/{db_id}.sqlite",
                         "prediction": "predicted"},
            "with_values": False
        }
    ]

    for evaluation_file in evaluation_files:
        add_metrics_in_file_with_predictions(file_path=evaluation_file["path"], mappings=evaluation_file["mappings"],
                                             calc_exec=False if "with_values" in evaluation_file \
                                                                and evaluation_file["with_values"] is False else True)

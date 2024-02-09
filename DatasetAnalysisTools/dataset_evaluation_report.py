import json

from DatasetAnalysisTools.DatasetInfo.dataset_evaluation import DatasetEvaluationInfo
from DatasetAnalysisTools.utils.create_plots import create_figure
from DatasetAnalysisTools.utils.datasets_getters import spider_getter


def create_generic_info_report(dataset_evaluation_info: DatasetEvaluationInfo, save_dir: str) -> None:
    """
    Creates a csv file containing the following information
        - the total errors of each model in every metric

    and the following plots:
        - Errors per metric.png
        - General errors.png
        - Structural errors per metric.png
        - Operators match errors.png
        - Variables match errors.png

    Args:
        dataset_evaluation_info (DatasetEvaluationInfo): The evaluation of a dataset
        save_dir (str): The directory in which the report will be saved.

    """

    dataset_evaluation_info.totalErrors().to_csv(f"{save_dir}/general_evaluation_information.csv")

    metrics = ["EM", "EX", "PM", "SM", "OM", "VM"]

    # Create a plot with the total errors of each metric
    totalErrors = dataset_evaluation_info.totalErrors()
    # Create a plot for every group of metrics
    create_figure(data=totalErrors[totalErrors["Metric"].isin(metrics)],
                  save_path=f"{save_dir}/Total errors", x="Metric",
                  y="% Errors",
                  hue="Model",
                  height=5, top=0.8, aspect=None, ncol=len(totalErrors["Model"].unique()), palette="rocket")

    # Create a plot with the errors per structural combinations
    structuralCombinationsErrors = dataset_evaluation_info.structural_categories_errors_df()

    create_figure(data=structuralCombinationsErrors[structuralCombinationsErrors["Metric"].isin(metrics)], row="Metric",
                  save_path=f"{save_dir}/Structural Categorization errors",
                  x="structural_category_name",
                  y="% Errors", hue="Model", height=5, top=0.9, aspect=None,
                  ncol=len(structuralCombinationsErrors["Model"].unique()), palette="rocket")

    create_figure(data=structuralCombinationsErrors[structuralCombinationsErrors["Metric"].isin(metrics)], row="Metric",
                  save_path=f"{save_dir}/Structural Categorization errors normalized",
                  x="structural_category_name",
                  y="% Errors Normalized", hue="Model", height=5, top=0.9, aspect=None,
                  ncol=len(structuralCombinationsErrors["Model"].unique()), palette="rocket")

    # Create a plot with the errors per operator type combination
    operatorTypesCombinationsErrors = dataset_evaluation_info.operatorTypes_categories_errors_df()

    create_figure(data=operatorTypesCombinationsErrors[operatorTypesCombinationsErrors["Metric"].isin(metrics)],
                  row="Metric",
                  save_path=f"{save_dir}/"
                            f"Operator Types Combinations Categorization errors",
                  x="operatorTypes_category_name",
                  y="% Errors", hue="Model", height=5, top=0.9, aspect=None,
                  ncol=len(operatorTypesCombinationsErrors["Model"].unique()), palette="rocket")

    create_figure(data=operatorTypesCombinationsErrors[operatorTypesCombinationsErrors["Metric"].isin(metrics)],
                  row="Metric",
                  save_path=f"{save_dir}"
                            f"/Operator Types Combinations Categorization errors normalized",
                  x="operatorTypes_category_name",
                  y="% Errors Normalized", hue="Model", height=5, top=0.9, aspect=None,
                  ncol=len(operatorTypesCombinationsErrors["Model"].unique()), palette="rocket")


def create_dataset_analysis_report(dataset: DatasetEvaluationInfo, save_dir: str) -> None:
    """
    Creates a dataset evaluation report. It creates .json files and visualizations of statistic
    through plots.

    Args:
        dataset (DatasetEvaluationInfo): THe information of model's evaluations in a dataset
        save_dir (str): The path of the directory in which the report files will be created.

    """

    # Create a report with the most generic information about the evaluation
    create_generic_info_report(dataset_evaluation_info=dataset, save_dir=save_dir)


if __name__ == "__main__":

    # Evaluation of other datasets
    evaluation_configs = [
        {
            "save_dir": "storage/datasets_evaluation_reports/models_from_others/spider",
            "dataset_info": {"name": "spider", "dataset": spider_getter()["dev"]},
            "predictions_info": {
                "T5-base": {
                    "predictions_file": "storage/evaluation_files/spider/models_from_others/t5_base_eval.json",
                    "args": {"with_values": True},
                    "predictions_key": "predicted"
                },
                "T5-large": {
                    "predictions_file": "storage/evaluation_files/spider/models_from_others/t5_large_eval.json",
                    "args": {"with_values": True},
                    "predictions_key": "predicted"
                },
                "T5-large_lm100k": {
                    "predictions_file": "storage/evaluation_files/spider/models_from_others/t5_large_lm100k_eval.json",
                    "args": {"with_values": True},
                    "predictions_key": "predicted"
                },
                "T5-3B": {
                    "predictions_file": "storage/evaluation_files/spider/models_from_others/t5_3b_eval.json",
                    "args": {"with_values": True},
                    "predictions_key": "predicted"
                },
                "RATSQL": {
                    "predictions_file": "storage/evaluation_files/spider/models_from_others/bart_run_1_true_1-step41000.eval",
                    "args": {"with_values": False},
                    "predictions_key": "predicted"
                },
                "DINSQL": {
                    "predictions_file": "storage/evaluation_files/spider/models_from_others/DIN-SQL.json",
                    "args": {"with_values": True},
                    "predictions_key": "predicted"
                },
            }
        }
    ]

    for evaluation_config in evaluation_configs:

        models = {}
        data_predictions = {}
        data_metrics = {}
        # For every model
        for model_name, model_info in evaluation_config["predictions_info"].items():
            with open(model_info["predictions_file"], "r") as f:
                model_predictions = json.load(f)
                if "per_item" in model_predictions:
                    model_predictions = model_predictions["per_item"]

            data_predictions[model_name] = []
            data_metrics[model_name] = []
            for prediction_info in model_predictions:
                data_predictions[model_name].append(prediction_info[model_info["predictions_key"]])
                data_metrics[model_name].append(prediction_info["metrics"])

            models[model_name] = model_info["args"]

        data_predictions = [dict(zip(data_predictions.keys(), values)) for values in zip(*data_predictions.values())]
        data_metrics = [dict(zip(data_metrics.keys(), values)) for values in zip(*data_metrics.values())]

        dataset_name = evaluation_config["dataset_info"]["name"]
        dataset = evaluation_config["dataset_info"]["dataset"]
        # add predictions to dataset
        for datapoint, predictions, metrics in zip(dataset, data_predictions, data_metrics):
            datapoint["predictions"] = predictions
            datapoint["metrics"] = metrics

        dataset_evaluation_info = DatasetEvaluationInfo(datasetName=dataset_name,
                                                        models=models, dataset=dataset)

        create_dataset_analysis_report(dataset_evaluation_info,
                                       save_dir=evaluation_config["save_dir"])

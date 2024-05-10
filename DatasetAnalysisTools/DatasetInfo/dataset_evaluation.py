from typing import Union

from tqdm import tqdm
import pandas as pd
import json

from DatasetAnalysisTools.DatasetInfo.dataset_info import DatasetInfo

PREDICTION_PARSING_ERRORS_OUTPUT_FILE = "storage/prediction_parsing_errors.json"


class DatasetEvaluationInfo(DatasetInfo):
    """
    Used to store the information about the metrics of a set of models in a dataset
    and provide methods for error statistics.

    """

    def __init__(self, datasetName: str, models: dict[str, dict], dataset: list[dict],
                 metrics: list[dict] = None):
        """
        Args:
            datasetName (str): The name of the evaluated dataset
            models (dict[str, dict]): The names of the models that are included in the evaluation with
                                      information about each model (e.g., with_values: boolean)
            dataset (list[dict]): A list with the datapoints of the evaluated dataset. Each datapoint must contain:
                 - sql_query (str): The ground truth sql_query
                 - question (str): The natural language question corresponding to the sql query
                 - db_id (str): The id of the database upon which the query is made

                 - predictions (dict[str, str]): A dictionary with the prediction of each model
                 - metrics (dict[str, dict[str, float]): A dictionary with the metric results of each model
                        (dict[<model_name>, dict[<metric_name>, <metric_value>]])
            and optionally the
                 - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
                               upon which the queries are made.

        !!! All datapoints must use the same models and have results in the same metrics.
        """

        super().__init__(datasetName, dataset, store_sql_queries_info=True, ignore_info=["questions"])

        # Initialize the evaluated models
        self.models = models
        # Initialize the metrics
        self.metrics = list(dataset[0]["metrics"][list(self.models.keys())[0]].keys())

        if len(self.metrics) == 0:
            raise Exception("At least one metric must be given!")

        # Create the dictionary with the evaluation results
        self._evaluationDict = self._initialize_evaluationDf()

        self._ignored_predictions_per_model = {m: 0 for m in self.models}
        self._prediction_parsing_errors_per_model = {m: [] for m in self.models}

        # For every sql query
        for _, row in tqdm(self.queries_info_df[self.queries_info_df["current_depth"] == -1].iterrows(),
                           desc="Creating evaluation report..."):
            try:
                self._append_queries_evaluation_rows(index=row["index"],
                                                     sql_query=row["sql_query"],
                                                     predictions=dataset[row["index"]]["predictions"],
                                                     metrics=dataset[row["index"]]["metrics"]
                                                     )
            except Exception as e:
                pass

        print(f"# Errors from parsing.................................................")
        print(", ".join([f"{model}: {num}" for model, num in self._ignored_predictions_per_model.items()]))
        print(".......................................................................")

        # Save to a file the errors during parsing
        with open(f"{PREDICTION_PARSING_ERRORS_OUTPUT_FILE}", "w") as error_file:
            error_file.write(json.dumps(self._prediction_parsing_errors_per_model))

        self.evaluationDf = pd.DataFrame(self._evaluationDict)
        self._evaluationDict = None

    @staticmethod
    def _initialize_evaluationDf() -> dict:
        """
        Defines the columns of the stored evaluation info
        Return (dict): Dictionary with keys: names of evaluation info to be stored, values: empty list
        """
        return {"index": [], "model": [], "prediction": [], "metric name": [], "metric value": []}

    def _append_queries_evaluation_rows(self, index: int, sql_query: str,
                                        predictions: dict[str, str], metrics: dict[str, dict[str, float]]) \
            -> None:
        """
           Args:
               index (int): The index of the query in the dataset
               sql_query (str): The sql query
               # db_schema (dict[str, list[str]): the schema of the database upon the query is done
               predictions (dict): {<model_name>: <prediction of the model>}
               metrics (dict[str, dict[str, float]]): {<model_name>: {<metric_name>: <metric_value>}}
        """

        for model, prediction in predictions.items():
            for metric_name, metric_value in metrics[model].items():
                self._evaluationDict["index"].append(index)
                self._evaluationDict["model"].append(model)
                self._evaluationDict["prediction"].append(prediction)
                self._evaluationDict["metric name"].append(metric_name)
                self._evaluationDict["metric value"].append(metric_value)

    def _errors(self, indexes: list[int] = None) -> pd.DataFrame:
        """
        Calculates the percentages of errors for each metric.
        """
        data = self.evaluationDf if indexes is None else self.evaluationDf[
            self.evaluationDf["index"].isin(indexes)]
        data_num = len(data["index"].unique())

        queries_num = self.queries_info_df["index"].nunique()

        totalErrors = {"Metric": [], "Model": [], "% Errors": [], "% Errors Normalized": []}
        # For every metric
        for metric_name in data["metric name"].unique():
            # For every model
            for model in data["model"].unique():
                totalErrors["Metric"].append(metric_name)
                totalErrors["Model"].append(model)
                model_metric_evaluation = data[(data["model"] == model) & (data["metric name"] == metric_name)]
                errors_num = model_metric_evaluation[((model_metric_evaluation["metric value"] != 1) |
                                                      (model_metric_evaluation["metric value"] == False))].shape[0]
                totalErrors["% Errors"].append(float(errors_num) / queries_num * 100)
                totalErrors["% Errors Normalized"].append(float(errors_num) / data_num * 100)

        return pd.DataFrame(totalErrors)

    def structural_categories_errors_df(self, number=10):
        structureTypes = self.structural_categories_df(categ_num=number, max_depth=-1, keep_indexes=True)
        structureTypesErrors = []
        for _, row in structureTypes.iterrows():
            structureTypeErrors = self._errors(row["Indexes"])
            structureTypeErrors["structural_category_name"] = row["structural_category_name"]
            structureTypesErrors.append(structureTypeErrors)

        return pd.concat(structureTypesErrors)

    def operatorTypes_categories_errors_df(self, number=10):
        operatorTypes = self.operatorTypes_categories_df(categ_num=number, max_depth=-1, keep_indexes=True)
        operatorTypesErrors = []
        for _, row in operatorTypes.iterrows():
            operatorTypeErrors = self._errors(row["Indexes"])
            operatorTypeErrors["operatorTypes_category_name"] = row["operatorTypes_category_name"]
            operatorTypesErrors.append(operatorTypeErrors)

        return pd.concat(operatorTypesErrors)

    def totalErrors(self):
        totalErrors = self._errors()
        totalErrors["% Success"] = 100 - totalErrors["% Errors"]
        return totalErrors

    def parsingErrors(self):
        return pd.DataFrame(data={"Model": self._ignored_predictions_per_model.keys(),
                                  "# Errors": self._ignored_predictions_per_model.values()})

    def models_differences(self, metric: str) -> pd.DataFrame:
        """Returns a dataframe with the sql_queries for which the models had correct and wrong predictions."""

        differences = {"sql_query": [], "correct_models": [], "predictions": []}

        predictions_per_datapoint = self.evaluationDf.groupby(["index"])

        for index, predictions in predictions_per_datapoint:
            metric_predictions = predictions[predictions["metric name"] == metric]
            if metric_predictions["metric value"].nunique() > 1:
                differences["sql_query"].append(self.queries_info_df[(self.queries_info_df["index"] == index) &
                                                                     (self.queries_info_df["current_depth"] == -1)]["sql_query"].values[0])
                differences["correct_models"].append(metric_predictions[metric_predictions["metric value"] == 1]["model"].values.tolist())
                differences["predictions"].append({row["model"]: row["prediction"] for _, row in metric_predictions.iterrows()})

        return pd.DataFrame(differences)

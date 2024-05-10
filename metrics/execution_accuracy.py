"""Spider execution accuracy metric."""
from datasets import Features, Value
from evaluate import EvaluationModule, EvaluationModuleInfo
from tqdm import tqdm
from loguru import logger
import signal

from metrics.utils.mysql_execution_accuracy_evaluator import mysql_exec_evaluator
# from third_party.test_suite import evaluation as spider_evaluation
from metrics.utils import metric_calculator as spider_evaluation


class SpiderExecutionAccuracyMetric(EvaluationModule):

    def _info(self) -> EvaluationModuleInfo:
        return EvaluationModuleInfo(description="Spider Execution Accuracy",
                                    citation="",
                                    features=Features(
                                        {
                                            "predictions": Value("string", id="sequence"),
                                            "references": {
                                                "sql_query": Value("string", id="sequence"),
                                                "db_id": Value("string", id="sequence"),
                                                "db_path": Value("string", id="sequence"),
                                            }
                                        }
                                    ),
                                    module_name="spider_execution_accuracy")

    def _compute(self, predictions=None, references=None, return_errors: bool = False, **kwargs):

        syntax_errors = []
        timeout_errors = []

        # Calculate execution accuracy with the evaluator of spider
        sqlite_evaluator = spider_evaluation.MetricCalculator(etype="exec",
                                                              plug_value=False,
                                                              keep_distinct=True,
                                                              progress_bar_for_each_datapoint=False
                                                              )

        evaluation_per_pair = []
        for i, (prediction, reference) in tqdm(enumerate(zip(predictions, references)),
                                               desc="Calculating execution accuracy..."):

            if prediction is None:
                evaluation_per_pair.append({"exec": 0})
            elif reference["db_path"].endswith(".sqlite"):
                try:
                    evaluation = sqlite_evaluator.evaluate_one(db_name=reference["db_id"],
                                                               gold=reference["sql_query"],
                                                               predicted=prediction,
                                                               db_path=reference["db_path"],
                                                               turn_scores={"exec": [], "exact": []},
                                                               idx=0)
                    evaluation_per_pair.append({"exec": evaluation["exec"]})
                except TimeoutError:
                    timeout_errors.append({"query": reference["sql_query"], "prediction": prediction,
                                           "db_path": reference["db_path"]})
                except Exception as e:
                    syntax_errors.append({"query": reference["sql_query"], "prediction": prediction,
                                          "db_path": reference["db_path"], "error": str(e)})
                    evaluation_per_pair.append({"exec": 0})
            else:
                try:
                    evaluation = mysql_exec_evaluator(db_name=reference["db_id"],
                                                      gold=reference["sql_query"],
                                                      predicted=prediction)
                    evaluation_per_pair.append(evaluation)
                except TimeoutError as e:
                    timeout_errors.append({"query": reference["sql_query"], "prediction": prediction,
                                           "db_path": reference["db_path"]})
                except SyntaxError as e:
                    syntax_errors.append({"query": reference["sql_query"], "prediction": prediction,
                                          "db_path": reference["db_path"], "error": str(e)})
                    evaluation_per_pair.append({"exec": 0})

        if len(syntax_errors):
            logger.warning(
                f"There were {len(syntax_errors)} pairs with syntax errors! The execution match was set to 0 "
                f"in these pairs.")

        if len(timeout_errors):
            logger.warning(
                f"There were {len(timeout_errors)} pairs with timeout errors! The execution match was set to 0 "
                f"in these pairs.")

        def add_error_type(errors, type_name):
            for error in errors:
                error["type"]: type_name
            return errors

        if "not_aggregated" in kwargs and kwargs["not_aggregated"]:
            if return_errors:
                errors = add_error_type(syntax_errors, "syntax error") + \
                         add_error_type(timeout_errors, "timeout error")
                return {"results": evaluation_per_pair, "errors": errors}
            else:
                return evaluation_per_pair
        else:
            return {"exec": sum([evaluation["exec"] for evaluation in evaluation_per_pair]) / len(evaluation_per_pair)}

"""Test-suite execution accuracy metric."""
from datasets import Features, Value
from evaluate import EvaluationModule, EvaluationModuleInfo
from tqdm import tqdm

from third_party.test_suite import evaluation as spider_evaluation


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

    def _compute(self, predictions=None, references=None, **kwargs):

        # Calculate execution accuracy with the evaluator of spider
        evaluator = spider_evaluation.Evaluator(db_dir="",
                                                kmaps=None,
                                                etype="exec",
                                                plug_value=False,
                                                keep_distinct=True,
                                                progress_bar_for_each_datapoint=False
                                                )
        evaluation_per_pair = []
        for prediction, reference in tqdm(zip(predictions, references)):
            if prediction is None:
                evaluation_per_pair.append({"exec": 0})
            else:
                try:
                    evaluation = evaluator.evaluate_one(db_name=reference["db_id"],
                                                        gold=reference["sql_query"],
                                                        predicted=prediction,
                                                        db_path=reference["db_path"],
                                                        turn_scores={"exec": [], "exact": []},
                                                        idx=0)
                    evaluation_per_pair.append({"exec": evaluation["exec"]})
                except Exception as e:
                    evaluation_per_pair.append({"exec": 0})

        evaluator.finalize()

        if "not_aggregated" in kwargs and kwargs["not_aggregated"]:
            return evaluation_per_pair
        else:
            return {"exec": evaluator.scores["all"]["exec"]}

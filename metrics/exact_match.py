"""Spider exact match metric."""
from datasets import Features, Value
from evaluate import EvaluationModule, EvaluationModuleInfo
from loguru import logger
from tqdm import tqdm

from metrics.utils import metric_calculator as spider_evaluation
from metrics.utils.normalize_for_exact_match_parsing import normalize_for_exact_match_parsing


def evaluation_in_exact_match_failure():
    return {
        "hardness": "NA",
        "exact": 0,
        "partial": "NA"
    }


class SpiderExactMatchMetric(EvaluationModule):

    def _info(self) -> EvaluationModuleInfo:
        return EvaluationModuleInfo(description="Spider Exact Match",
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
                                    module_name="spider_exact_match")

    def _compute(self, predictions=None, references=None, return_errors: bool = False, **kwargs):

        evaluator = spider_evaluation.MetricCalculator(
            etype="match",
            plug_value=False,
            keep_distinct=True,
            progress_bar_for_each_datapoint=False,
        )

        parsing_errors = []

        evaluation_per_pair = []
        for prediction, reference in tqdm(zip(predictions, references), desc="Calculating exact match..."):
            try:
                evaluation = evaluator.evaluate_one(db_name=reference["db_id"],
                                                    gold=normalize_for_exact_match_parsing(reference["sql_query"]),
                                                    predicted=normalize_for_exact_match_parsing(prediction),
                                                    db_path=reference["db_path"],
                                                    turn_scores={"exec": [], "exact": []},
                                                    idx=0)
            except Exception as e:  # If there is a problem during parsing
                if len(prediction):
                    parsing_errors.append({"query": reference["sql_query"], "prediction": prediction,
                                          "db_path": reference["db_path"], "error": str(e)})
                evaluation = evaluation_in_exact_match_failure()

            evaluation_per_pair.append({k: v for k, v in evaluation.items() if k in ("hardness", "exact", "partial")})

        if len(parsing_errors):
            logger.warning(f"There were {len(parsing_errors)} pairs with syntax errors! The exact match was set to 0 "
                           f"in these pairs")

        if "not_aggregated" in kwargs and kwargs["not_aggregated"]:
            if return_errors:
                return {"results": evaluation_per_pair, "errors": parsing_errors}
            else:
                return evaluation_per_pair
        else:
            return {
                "exact": sum([evaluation["exact"] for evaluation in evaluation_per_pair]) / len(evaluation_per_pair)}

"""Test-suite exact match metric."""
from datasets import Features, Value
from evaluate import EvaluationModule, EvaluationModuleInfo

from DatasetAnalysisTools.DatabaseInfo.database_info import DatabaseInfo
from third_party.test_suite import evaluation as spider_evaluation


def evaluation_in_exact_match_failure():
    return {
                "hardness": "NA",
                "exact": 0,
                "partial":  "NA"
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

    def _compute(self, predictions=None, references=None, **kwargs):

        foreign_key_maps = dict()
        dbs_info = {}
        for reference in references:
            if reference['db_path'] is not None:
                dbs_info[reference["db_id"]] = DatabaseInfo(reference['db_path']).schema_as_dict()
            else:
                raise Exception("No database info are provided!")

        # Create foreign-primary key mapping
        for db_id, db_info in dbs_info.items():
            if db_id not in foreign_key_maps:
                foreign_key_maps[db_id] = spider_evaluation.build_foreign_key_map(
                    {
                        "table_names_original": db_info["table_names"],
                        "column_names_original": db_info["column_names"],
                        "foreign_keys": db_info["foreign_primary_keys"],
                    }
                )

        # Calculate exact match with the evaluator of spider
        evaluator = spider_evaluation.Evaluator(db_dir="",
                                                kmaps=foreign_key_maps,
                                                etype="match",
                                                plug_value=True,
                                                keep_distinct=True,
                                                progress_bar_for_each_datapoint=False)
        evaluation_per_pair = []
        for prediction, reference in zip(predictions, references):
            try:
                evaluation = evaluator.evaluate_one(db_name=reference["db_id"],
                                                    gold=reference["sql_query"],
                                                    predicted=prediction,
                                                    db_path=reference["db_path"],
                                                    turn_scores={"exec": [], "exact": []},
                                                    idx=0)
            except Exception as e:  # If there is a problem during parsing
                evaluation = evaluation_in_exact_match_failure()
            evaluation_per_pair.append({k: v for k, v in evaluation.items() if k in ("hardness", "exact", "partial")})

        evaluator.finalize()

        if "not_aggregated" in kwargs and kwargs["not_aggregated"]:
            return evaluation_per_pair
        else:
            return {"exact": evaluator.scores["all"]["exact"]}

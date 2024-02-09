# code based on https://github.com/taoyds/test-suite-sql-eval/

from DatasetAnalysisTools.DatabaseInfo.database_info import DatabaseInfo
from third_party.test_suite.evaluation import PARTIAL_TYPES
from third_party.test_suite.evaluation import Evaluator as TestSuiteEvaluator
from third_party.test_suite.evaluation import (
    build_foreign_key_map,
    build_valid_col_units,
    rebuild_sql_col,
    rebuild_sql_val,
)
from third_party.test_suite.exec_eval import eval_exec_match
from third_party.test_suite.process_sql import Schema as TestSuiteSchema
from third_party.test_suite.process_sql import get_schema, get_sql


class MetricCalculator(TestSuiteEvaluator):
    def __init__(
        self, etype, plug_value, keep_distinct, progress_bar_for_each_datapoint
    ):
        super().__init__(
            None, {}, etype, plug_value, keep_distinct, progress_bar_for_each_datapoint
        )

    def evaluate_one(self, db_path, gold, predicted, turn_scores={"exec": [], "exact": []}, idx=0):
        # CHANGE FROM TEST SUITE OFFICIAL EVALUATOR.....................................................................
        if db_path not in self.db_paths:
            self.db_paths[db_path] = db_path
            self.schemas[db_path] = TestSuiteSchema(get_schema(db_path))

            db_info = DatabaseInfo(db_path)

            self.kmaps[db_path] = build_foreign_key_map(
                {
                    "table_names_original": db_info.table_names,
                    "column_names_original": db_info.column_names,
                    "foreign_keys": db_info.foreign_primary_keys,
                }
            )
        # ..............................................................................................................

        if idx > 3:
            idx = "> 4"
        else:
            idx += 1
        turn_id = "turn " + str(idx)

        self.scores[turn_id]["count"] += 1
        self.scores["all"]["count"] += 1

        if self.etype in ["all", "match"]:
            schema = self.schemas[db_path]
            g_sql = get_sql(schema, gold)
            hardness = self.eval_hardness(g_sql)
            self.scores[hardness]["count"] += 1

            try:
                p_sql = get_sql(schema, predicted)
            except:
                # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
                p_sql = {
                    "except": None,
                    "from": {"conds": [], "table_units": []},
                    "groupBy": [],
                    "having": [],
                    "intersect": None,
                    "limit": None,
                    "orderBy": [],
                    "select": [False, []],
                    "union": None,
                    "where": [],
                }

        if self.etype in ["all", "exec"]:
            exec_score = eval_exec_match(
                db=self.db_paths[db_path],
                p_str=predicted,
                g_str=gold,
                plug_value=self.plug_value,
                keep_distinct=self.keep_distinct,
                progress_bar_for_each_datapoint=self.progress_bar_for_each_datapoint,
            )
            if exec_score:
                if self.etype == "all":
                    self.scores[hardness]["exec"] += 1
                self.scores[turn_id]["exec"] += 1
                self.scores["all"]["exec"] += 1
                turn_scores["exec"].append(1)
            else:
                turn_scores["exec"].append(0)

        if self.etype in ["all", "match"]:
            # rebuild sql for value evaluation
            kmap = self.kmaps[db_path]
            g_valid_col_units = build_valid_col_units(
                g_sql["from"]["table_units"], schema
            )
            g_sql = rebuild_sql_val(g_sql)
            g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
            p_valid_col_units = build_valid_col_units(
                p_sql["from"]["table_units"], schema
            )
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
            partial_scores = self.eval_partial_match(p_sql, g_sql)
            exact_score = self.eval_exact_match(p_sql, g_sql, partial_scores)
            if exact_score == 0:
                turn_scores["exact"].append(0)
                print("{} pred: {}".format(hardness, predicted))
                print("{} gold: {}".format(hardness, gold))
                print("")
            else:
                turn_scores["exact"].append(1)
            self.scores[turn_id]["exact"] += exact_score
            self.scores[hardness]["exact"] += exact_score
            self.scores["all"]["exact"] += exact_score
            for type_ in PARTIAL_TYPES:
                if partial_scores[type_]["pred_total"] > 0:
                    self.scores[hardness]["partial"][type_]["acc"] += partial_scores[
                        type_
                    ]["acc"]
                    self.scores[hardness]["partial"][type_]["acc_count"] += 1
                if partial_scores[type_]["label_total"] > 0:
                    self.scores[hardness]["partial"][type_]["rec"] += partial_scores[
                        type_
                    ]["rec"]
                    self.scores[hardness]["partial"][type_]["rec_count"] += 1
                self.scores[hardness]["partial"][type_]["f1"] += partial_scores[type_][
                    "f1"
                ]
                if partial_scores[type_]["pred_total"] > 0:
                    self.scores["all"]["partial"][type_]["acc"] += partial_scores[
                        type_
                    ]["acc"]
                    self.scores["all"]["partial"][type_]["acc_count"] += 1
                if partial_scores[type_]["label_total"] > 0:
                    self.scores["all"]["partial"][type_]["rec"] += partial_scores[
                        type_
                    ]["rec"]
                    self.scores["all"]["partial"][type_]["rec_count"] += 1
                self.scores["all"]["partial"][type_]["f1"] += partial_scores[type_][
                    "f1"
                ]

        result = {
            "predictSQL": predicted,
            "goldSQL": gold,
        }
        if self.etype in ["all", "match"]:
            result.update(
                {
                    "hardness": hardness,
                    "exact": exact_score,
                    "partial": partial_scores,
                }
            )
        if self.etype in ["all", "exec"]:
            result["exec"] = exec_score
        return result

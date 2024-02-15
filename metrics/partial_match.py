from abc import ABC, abstractmethod
from typing import Any
import signal
import mo_parsing
import pandas as pd
from datasets import Features, Value
from evaluate import EvaluationModule, EvaluationModuleInfo

from DatasetAnalysisTools.QueryInfo.query_extractor.mo_sql_parser_extractor import MoQueryExtractor
from DatasetAnalysisTools.QueryInfo.query_info import QueryInfo, get_subqueries_per_depth, CLAUSES, VARIABLE_TYPES, \
    OPERATOR_TYPES


def timeout_handler(signum, frame):
    raise TimeoutError()


class ComponentsMatch(ABC):
    """Calculates the match for a set of components."""

    def __init__(self, reference: QueryInfo, prediction: QueryInfo):
        self.matches = self._calc(reference, prediction)

    @classmethod
    @abstractmethod
    def components(cls) -> list[str]:
        pass

    @abstractmethod
    def _extract_elements(self, query: QueryInfo) -> list[str]:
        """Returns a list with the elements existing in the query regarding the components."""
        pass

    @abstractmethod
    def _get_component_instances(self, component: str, elements: list[str]) -> list[str]:
        """Returns a sublist of the elements that contains only the component instances."""
        pass

    @staticmethod
    def _get_max_subqueries_per_depth(ref_subqueries: dict[int, list[QueryInfo]],
                                      pred_subqueries: dict[int, list[QueryInfo]]) -> list[int]:
        # Calculate max depth
        max_depth = max(max(list(ref_subqueries.keys())), max(list(pred_subqueries.keys())))

        max_subqueries_per_depth = []
        for depth in range(max_depth + 1):
            if len(ref_subqueries) > depth and len(pred_subqueries) > depth:
                max_subqueries_per_depth.append(max(len(ref_subqueries[depth]), len(pred_subqueries[depth])))
            elif len(ref_subqueries) <= depth:
                max_subqueries_per_depth.append(len(pred_subqueries[depth]))
            else:  # len(pred_subqueries) <= depth:
                max_subqueries_per_depth.append(len(ref_subqueries[depth]))
        return max_subqueries_per_depth

    @staticmethod
    def _jaccard_similarity(a: list[str], b: list[str]) -> float:
        a = set(a)
        b = set(b)
        intersect_len = len(a.intersection(b))
        union_len = len(a.union(b))
        if union_len > 0:
            return float(intersect_len) / union_len
        else:
            return 1

    @staticmethod
    def _index_exist(data: dict[int, list[Any]], index1: int, index2: int) -> bool:
        """Returns True if in the given list there is the dimension data[index1][index2], False otherwise."""
        if index1 in data and len(data[index1]) > index2:
            return True
        else:
            return False

    def _similarity(self, ref_elements: list[str], pred_elements: list[str]) -> float:
        """Returns the similarity score of 2 lists."""
        return self._jaccard_similarity(ref_elements, pred_elements)

    def _get_components_similarities(self, ref_elements: list[str], pred_elements: list[str]) \
            -> dict[str, float]:
        """
        Returns the similarity of every component
        Args:
            ref_elements (list[str]): The list of the elements existing in the referenced subquery
            pred_elements (list[str]): The list of the elements existing in the predicted subquery

        Returns (dict[str, float]): A dictionary with the match value of every component
        """

        matches = {}

        # For every component
        for component in self.components():
            # Compute the component's similarity
            similarity = self._similarity(
                self._get_component_instances(component, ref_elements),
                self._get_component_instances(component, pred_elements)
            )
            matches[component] = similarity

        return matches

    def _calc(self, reference: QueryInfo, prediction: QueryInfo) -> dict[int, dict[str, float]]:
        """
        Calculates the component match of 2 queries.

        Returns (dict[dict]]): A dictionary with the total and each component match value in every depth.
        """

        # Get the subqueries of each depth
        ref_subqueries = get_subqueries_per_depth(reference)
        pred_subqueries = get_subqueries_per_depth(prediction)

        max_subqueries_per_depth = self._get_max_subqueries_per_depth(ref_subqueries, pred_subqueries)

        matches = {d: {} for d in range(len(max_subqueries_per_depth))}

        # For every depth
        for depth, max_subqueries_num in enumerate(max_subqueries_per_depth):

            depth_similarities = {k: [] for k in ["total"] + self.components()}
            # Compare the subqueries of the current depth
            for i in range(max_subqueries_num):

                # Check that the i-th subquery in the current depth exist both in the reference and in prediction
                ref_exist = self._index_exist(ref_subqueries, depth, i)
                pred_exist = self._index_exist(pred_subqueries, depth, i)

                # Extract the elements of each subqueries
                ref_elements = self._extract_elements(ref_subqueries[depth][i]) if ref_exist else []
                pred_elements = self._extract_elements(pred_subqueries[depth][i]) if pred_exist else []

                # Calculate the total components similarity of the 2 subqueries
                depth_similarities["total"].append(self._similarity(ref_elements, pred_elements))

                # Calculate the similarity of each component in the subquery
                for component, component_similarity in self._get_components_similarities(ref_elements,
                                                                                         pred_elements).items():
                    depth_similarities[component].append(component_similarity)

                # Calculate the total and components match for the current depth
                matches[depth]["total"] = sum(depth_similarities["total"]) / len(depth_similarities["total"])
                for component in self.components():
                    matches[depth][component] = sum(depth_similarities[component]) / len(depth_similarities[component])

        return matches

    def value(self, depth: int = None, component: str = None) -> float:
        """
        Returns the requested match value.
        Args:
            depth (int): The depth that will be considered in the match value returned. If None, the average of all
                depths will be returned.
            component (str): The name of the component that will be considered in the match value returned. If None,
                the total match will be returned.
        """

        if depth is None and component is None:
            # Return the total match for all depths
            return sum([depth_matches["total"] for _, depth_matches in self.matches.items()]) / len(self.matches)
        elif component is not None and depth is not None:
            return self.matches[depth][component]
        elif depth is not None:
            return self.matches[depth]["total"]
        else:  # component is not None
            return sum([depth_matches[component] for _, depth_matches in self.matches.items()]) / len(self.matches)


class StructuralMatch(ComponentsMatch):

    def __init__(self, reference: QueryInfo, prediction: QueryInfo):
        super().__init__(reference, prediction)

    @classmethod
    def components(cls) -> list[str]:
        return [clause.class_name() for clause in CLAUSES] + ["set_operator"]

    @classmethod
    def name(cls):
        return "structural_match"

    def _get_component_instances(self, component: str, elements: list[str]) -> list[str]:
        if component == "set_operator":
            return [c for c in elements if c.startswith("union") | c.startswith("intersect") | c.startswith("except")]
        else:
            return [c for c in elements if c.startswith(component)]

    def _extract_elements(self, query: QueryInfo) -> list[str]:

        elements = query.structural_components(shallow_search=True, with_names=True)

        if "orderby" in elements:
            # Append the orders in the order by in the element
            elements[elements.index("orderby")] = f"orderby_{'_'.join(query.orderByClause.orders())}"

        # If the query is a set operator
        if query.setOperator is not None:
            set_operator_queries = query.setOperator.queries
            elements = set_operator_queries[0].structural_components(shallow_search=True, with_names=True) + [query.setOperator.op]
            elements2 = set_operator_queries[1].structural_components(shallow_search=True, with_names=True)
            elements.extend([f"{element}_1" for element in elements2])

        return elements


class OperatorsMatch(ComponentsMatch):

    def __init__(self, reference: QueryInfo, prediction: QueryInfo):
        super().__init__(reference, prediction)

    @classmethod
    def components(cls) -> list[str]:
        comps = []
        for operator_type in OPERATOR_TYPES:
            comps.extend(operator_type.members())
        return comps

    @classmethod
    def name(cls):
        return "operators_match"

    def _get_component_instances(self, component: str, elements: list[str]) -> list[str]:
        return [e for e in elements if e.startswith(component)]

    def _extract_elements(self, query: QueryInfo) -> list[str]:
        operators_per_clause = query.operators(shallow_search=True, per_clause=True)

        elements = []
        for clause, operators in operators_per_clause.items():
            for operator in operators:
                num = sum(f"{operator}_{clause}" in e for e in elements)
                elements.append(f"{operator}_{clause}_{num}")

        return elements


class VariablesMatch(ComponentsMatch):

    def __init__(self, reference: QueryInfo, prediction: QueryInfo, with_values: bool = True):
        self.with_values = with_values
        super().__init__(reference, prediction)

    @classmethod
    def components(cls) -> list[str]:
        return VARIABLE_TYPES

    @classmethod
    def name(cls):
        return "variables_match"

    def _get_component_instances(self, component: str, elements: list[str]) -> list[str]:
        return [e for e in elements if e.startswith(component + "/")]

    def _extract_elements(self, query: QueryInfo) -> list[str]:

        variable_types = self.components() if self.with_values else self.components().remove("values")

        self._columns_equivalences = query.get_columns_equivalences()

        elements = []
        for variable_type in variable_types:
            query_variables = getattr(query, variable_type)(shallow_search=True) \
                if variable_type != "columns" else getattr(query, "columns_without_table_aliases")(shallow_search=True)
            for variable in query_variables:
                # append element <variable_type>/<variable>/<i>
                new_element = f"{variable_type}/{str(variable)}"
                elements.append(new_element + "/" + str(sum(new_element + "/" in e for e in elements)))
        return elements

    def _replace_equivalence(self, set1: set[str], set2: set[str]) -> (set[str], set[str], bool):

        if len(self._columns_equivalences) == 0:
            return set1, set2, False

        set1_column_difference = {var for var in set1.difference(set2) if "columns/" in var}
        set2_column_difference = {var for var in set2.difference(set1) if "columns/" in var}

        found_equivalence = False
        for column1 in sorted(set1_column_difference, reverse=True):
            _, column_name1, number1 = column1.split("/")
            # If this is not the first appearance of the column (must appear at least one for the equivalence to exist)
            # and there exist an equivalence column
            if int(number1) > 0 and column_name1 in self._columns_equivalences:
                # search for the equivalent
                for column2 in sorted(set2_column_difference):
                    _, column_name2, number2 = column2.split("/")
                    if int(number2) > 0 and column_name2 in self._columns_equivalences[column_name1]:
                        found_equivalence = True
                        set1.remove(column1)
                        # append element <variable_type>/<variable>/<i>
                        new_element = "columns/" + str(column_name2).lower()
                        set1.add(new_element + "/" + str(sum(new_element + "/" in e for e in set1)))
                        break
        return set1, set2, found_equivalence

    def _similarity(self, ref_elements: list[str], pred_elements: list[str]) -> float:

        ref_elements = set(ref_elements)
        pred_elements = set(pred_elements)

        equivalence_detected = True
        while equivalence_detected:
            ref_elements, pred_elements, equivalence_detected = self._replace_equivalence(ref_elements, pred_elements)

        return super()._similarity(list(ref_elements), list(pred_elements))


class PartialMatch:

    def __init__(self, reference: QueryInfo, prediction: QueryInfo, with_values: bool = True):
        """Returns the match for every component of the partial match"""
        self.structural_match = StructuralMatch(reference, prediction)
        self.operators_match = OperatorsMatch(reference, prediction)
        self.variables_match = VariablesMatch(reference, prediction, with_values)

    @classmethod
    def name(cls):
        return "partial_match"

    @classmethod
    def components(cls):
        return [StructuralMatch, OperatorsMatch, VariablesMatch]

    @classmethod
    def reported_values(cls, extensive: bool = True):
        if extensive is False:
            return ["PM", "SM", "OM", "VM"]
        else:
            return ["PM", "SM", "OM", "VM"] + \
                StructuralMatch.components() + OperatorsMatch.components() + VariablesMatch.components()

    def value(self, depth: int = None, extensive_report: bool = False) -> dict[str, float]:
        report = {
            "PM": (self.structural_match.value(depth=depth) + self.operators_match.value(
                depth=depth) + self.variables_match.value(depth=depth)) / 3,
            "SM": self.structural_match.value(depth),
            "OM": self.operators_match.value(depth),
            "VM": self.variables_match.value(depth)
        }

        if extensive_report:
            report.update({component: self.structural_match.value(component=component)
                           for component in StructuralMatch.components()
                           })
            report.update({component: self.operators_match.value(component=component)
                           for component in StructuralMatch.components()
                           })
            report.update({component: self.variables_match.value(component=component)
                           for component in StructuralMatch.components()
                           })
        return report


class PartialMatchMetric(EvaluationModule):

    def _info(self) -> EvaluationModuleInfo:
        return EvaluationModuleInfo(description="Partial Match",
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
                                    module_name="partial_match")

    @classmethod
    def reported_values(cls, extensive: bool = True):
        return PartialMatch.reported_values(extensive=extensive)

    def _compute(self, predictions=None, references=None, **kwargs):

        mo_sql_extractor = MoQueryExtractor()

        extensive_report = True if "extensive_report" in kwargs and kwargs["extensive_report"] is True else False
        evaluation_results = {m: [] for m in PartialMatch.reported_values(extensive=extensive_report)}

        original_sigalrm_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, handler=timeout_handler)

        for prediction, reference in zip(predictions, references):
            try:
                signal.alarm(10)
                # Parse the predicted and the referenced query
                prediction_query_info = mo_sql_extractor.extract(prediction)
                reference_query_info = mo_sql_extractor.extract(reference["sql_query"])
                signal.alarm(0)

                # Calculate the partial match
                partial_match = PartialMatch(reference=reference_query_info,
                                             prediction=prediction_query_info, with_values=True)

                # For each metric calculated in the partial match component
                for metric, value in partial_match.value().items():
                    evaluation_results[metric].append(value)
            except Exception as e:
                signal.alarm(0)
                # If the exception is not due to the mozilla parser
                if type(e) is not mo_parsing.exceptions.ParseException:
                    pass

                # Add zero value to all metrics for this pair
                for metric in evaluation_results.keys():
                    evaluation_results[metric].append(0)

        signal.signal(signal.SIGALRM, original_sigalrm_handler)

        evaluation_results = pd.DataFrame(evaluation_results)

        # If metrics values should be boolean (0: mismatch, 1: match)
        if "boolean" in kwargs and kwargs["boolean"]:
            evaluation_results = evaluation_results.applymap(lambda x: 0 if x != 1 else 1)

        # If results need to be reported for each pair
        if "not_aggregated" in kwargs and kwargs["not_aggregated"]:
            return evaluation_results.to_dict(orient="records")
        else:
            return {metric: avg for metric, avg in evaluation_results.mean().items()}

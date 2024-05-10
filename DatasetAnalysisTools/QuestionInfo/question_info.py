from typing import Union, Literal
import numpy as np
import spacy
import textstat
import language_tool_python
from wordfreq import word_frequency

from DatasetAnalysisTools.QuestionInfo.utils.ast_info import QuestionSyntaxTree
from DatasetAnalysisTools.QuestionInfo.utils.text_complexity import TextComplexityMeasures

nlp = spacy.load("en_core_web_sm")

text_complexity_measures = TextComplexityMeasures()


class QuestionInfo:

    def __init__(self, question: str, schema_elements: list[str] = None, sql_query_schema_elements: list[str] = None):

        self.question = question
        self.nlp_question = None

        if schema_elements is not None:
            self.schema_elements = [schema_element.lower() for schema_element in schema_elements]
        else:
            self.schema_elements = None
        self.schema_references = self._get_schema_references(schema_elements) if schema_elements is not None else None
        # The entities used in the corresponding sql query
        self.sql_query_schema_elements = [schema_element.lower() for schema_element in
                                          list(set(sql_query_schema_elements))] \
            if sql_query_schema_elements is not None else None

        self._question_syntax_tree = None

    @staticmethod
    def _process_schema_elements(schema_elements: list[str]) -> list[str]:

        schema_elements = [schema_element.replace("_", " ") for schema_element in schema_elements]

        # lemmatize the schema elements
        schema_elements = [" ".join([token.lemma_ for token in nlp(schema_element)]) for schema_element in
                           schema_elements]

        return schema_elements

    def _get_nlp_question(self):
        if self.nlp_question is None:
            self.nlp_question = nlp(self.question)

        return self.nlp_question

    def _get_schema_references(self, schema_elements: list[str]) -> Union[list[str], None]:
        """Returns a list with the referenced schema elements in the question."""

        if schema_elements is not None:
            # Split question into words and get their lemmas
            question_lemmatized = " ".join([token.lemma_ for token in self._get_nlp_question()]) + " "
            processed_schema_elements = self._process_schema_elements(self.schema_elements)

            schema_references_idx = []
            for i, processed_schema_elements in enumerate(processed_schema_elements):
                if f" {processed_schema_elements} " in question_lemmatized:
                    schema_references_idx.append(i)

            return np.take(self.schema_elements, schema_references_idx).tolist()
        else:
            return None

    def question_len(self, entity: Literal["word", "char"] = "word") -> int:
        """ Returns the length of the question. """
        if entity == "char":
            return len(self.question)
        else:
            return len([token for token in self._get_nlp_question() if not token.is_punct])

    def referenced_schema_elements(self) -> Union[int, None]:
        """ Returns the number of schema elements referenced in the question. """
        return len(self.schema_references) if self.schema_references is not None else None

    def referenced_schema_elements_percentage(self) -> Union[float, None]:
        """
        Returns the percentage of sql elements in the sql query that exist in the question.
        If the sql schema elements are not available then None is returned.
        """
        if self.sql_query_schema_elements is not None and self.schema_references is not None:
            return (len(set(self.schema_references).intersection(set(self.sql_query_schema_elements))) /
                    len(self.sql_query_schema_elements) * 100)
        else:
            return None

    def readability(self, metric: Literal["flesch_reading_ease", "mcalpine_eflaw"] = "flesch_reading_ease") -> float:
        """
        Returns the readability score of a document.

        Args:
            metric(Literal["flesch_reading_ease", "mcalpine_eflaw"]): The metric used to calculate the readability
            score.
        """

        return getattr(textstat, str(metric))(self.question)

    def rarity(self) -> float:
        """Returns the rarity score of the question."""
        return text_complexity_measures.pos_based_metrics(self.question, metrics=["rarity"])["rarity"]

    def lexical_density(self) -> float:
        """Returns the lexical density score of the question."""
        return text_complexity_measures.pos_based_metrics(self.question, metrics=["lexical_density"])["lexical_density"]

    def avg_dependency_distance(self) -> float:
        """Returns the average dependency distance of the nodes in the question's dependency tree."""
        return text_complexity_measures.dependency_based_metrics(self.question,
                                                                 metrics=["average dependency distance"])[
            "average dependency distance"]

    def depth(self) -> float:
        """Returns the depth of the question's dependency tree."""
        if self._question_syntax_tree is None:
            self._question_syntax_tree = QuestionSyntaxTree(self.question)

        return self._question_syntax_tree.depth

    def dependencies_num(self) -> int:
        """Returns the number of dependencies existing in the question's dependency tree."""
        if self._question_syntax_tree is None:
            self._question_syntax_tree = QuestionSyntaxTree(self.question)

        return self._question_syntax_tree.edges

    def grammar_errors_num(self):
        tool = language_tool_python.LanguageTool('en-US')

        matches = tool.check(self.question)

        tool.close()

        return len(matches)

    def question_words_freqs(self):

        doc = self._get_nlp_question()

        words_freqs = [word_frequency(token.text, "en", wordlist="large") for token in doc if
                       not token.is_punct and not token.like_num and not token.is_stop]

        return sum(words_freqs) / len(words_freqs)

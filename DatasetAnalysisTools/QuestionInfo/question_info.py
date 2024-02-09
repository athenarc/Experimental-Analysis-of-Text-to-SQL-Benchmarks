from typing import Union

import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")


class QuestionInfo:

    def __init__(self, question: str, schema_elements: list[str] = None, sql_query_schema_elements: list[str] = None):

        self.question = question
        if schema_elements is not None:
            self.schema_elements = [schema_element.lower() for schema_element in schema_elements]
        else:
            self.schema_elements = None
        self.schema_references = self._get_schema_references(schema_elements) if schema_elements is not None else None
        # The entities used in the corresponding sql query
        self.sql_query_schema_elements = [schema_element.lower() for schema_element in
                                          list(set(sql_query_schema_elements))] \
            if sql_query_schema_elements is not None else None

    @staticmethod
    def _process_schema_elements(schema_elements: list[str]) -> list[str]:

        schema_elements = [schema_element.replace("_", " ") for schema_element in schema_elements]

        # lemmatize the schema elements
        schema_elements = [" ".join([token.lemma_ for token in nlp(schema_element)]) for schema_element in schema_elements]

        return schema_elements

    def _get_schema_references(self, schema_elements: list[str]) -> Union[list[str], None]:
        """Returns a list with the referenced schema elements in the question."""

        if schema_elements is not None:
            # Split question into words and get their lemmas
            question_lemmatized = " ".join([token.lemma_ for token in nlp(self.question)]) + " "
            processed_schema_elements = self._process_schema_elements(self.schema_elements)

            schema_references_idx = []
            for i, processed_schema_elements in enumerate(processed_schema_elements):
                if f" {processed_schema_elements} " in question_lemmatized:
                    schema_references_idx.append(i)

            return np.take(self.schema_elements, schema_references_idx).tolist()
        else:
            return None

    def question_len(self) -> int:
        """ Returns the length of the question. """
        return len(self.question)

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

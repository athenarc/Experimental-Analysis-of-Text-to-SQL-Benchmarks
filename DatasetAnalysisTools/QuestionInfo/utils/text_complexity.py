"""All the methods required to use the methods of the textcomplexity package without argparse"""
import itertools
import os
from typing import Literal
import stanza
import importlib.resources
from stanza.utils.conll import CoNLL
import json
import functools
from loguru import logger

from textcomplexity import surface, pos, dependency
from textcomplexity.utils import conllu, misc
from textcomplexity.utils.text import Text

# stanza.download("en")

"code refactored from https://github.com/tsproisl/textcomplexity/blob/master/textcomplexity/cli.py"


def read_language_definition(filename):
    with importlib.resources.open_text("textcomplexity", filename) as f:
        ld = json.load(f)
    return ld["language"], set(ld["punctuation"]), set(ld["proper_names"]), set(ld["open_classes"]), set(
        [(t, f) for t, f in ld["most_common"]])


def rare_words(text, reference_frequency_list, open_tags_ex_names):
    assert len(
        open_tags_ex_names) > 0, "You need to define proper names and open word classes in the language definition file"
    content_words = [t for t in zip(text.tokens, text.tags) if t[1] in open_tags_ex_names]
    rare = [t for t in content_words if t not in reference_frequency_list]
    return len(rare)


class TextComplexityMeasures:

    def __init__(self, preset: Literal["lexical_core", "core", "extended_core", "all", "custom_preset"] = "all",
                 lang: Literal["de", "en"] = "en", input_format: Literal["conllu"] = "conllu",
                 window_size: int = 100, ignore_case: bool = False):
        if lang == "de":
            (self.language, self.punct_tags, self.name_tags, self.open_tags,
             self.reference_frequency_list) = read_language_definition("de.json")
            self.nlp = stanza.Pipeline("de")
        else:  # lang == "en"
            (self.language, self.punct_tags, self.name_tags, self.open_tags,
             self.reference_frequency_list) = read_language_definition("en.json")
            self.nlp = stanza.Pipeline("en")

        self.preset = preset
        self.input_format = input_format
        self.window_size = window_size
        self.ignore_case = ignore_case

        if self.ignore_case:
            self.reference_frequency_list = set(
                [(w.lower(), t) for w, t in self.reference_frequency_list])

        self._surfaced_based_metrics = {
            "type-token ratio": surface.type_token_ratio,
            "evenness": surface.evenness,
            "Guiraud's R": surface.guiraud_r,
            "Herdan's C": surface.herdan_c,
            "Dugast's k": surface.dugast_k,
            "Maas' a²": surface.maas_a2,
            "Dugast's U": surface.dugast_u,
            "Tuldava's LN": surface.tuldava_ln,
            "Brunet's W": surface.brunet_w,
            "CTTR": surface.cttr,
            "Summer's S": surface.summer_s,
            "Sichel's S": surface.sichel_s,
            "Michéa's M": surface.michea_m,
            "Honoré's H": surface.honore_h,
            "entropy": surface.entropy,
            "Jarvis's evenness": surface.jarvis_evenness,
            "Yule's K": surface.yule_k,
            "Simpson's D": surface.simpson_d,
            "Herdan's Vm": surface.herdan_vm,
            "HD-D": surface.hdd,
            "average token length": surface.average_token_length,
            "Orlov's Z": surface.orlov_z,
            "Gini-based dispersion": functools.partial(surface.gini_based_dispersion, exclude_hapaxes=True),
            "evenness-based dispersion": functools.partial(surface.evenness_based_dispersion, exclude_hapaxes=True),
        }

        self._pos_based_metrics = {
            "lexical_density": functools.partial(pos.lexical_density, open_tags=self.open_tags),
            "rarity": functools.partial(pos.rarity, reference_frequency_list=self.reference_frequency_list,
                                        open_tags_ex_names=(self.open_tags - self.name_tags)),
            "rare_words": functools.partial(rare_words, reference_frequency_list=self.reference_frequency_list,
                                            open_tags_ex_names=(self.open_tags - self.name_tags))
        }

        self._dependency_based_metrics = {
            "average dependency distance": dependency.average_dependency_distance,
            "closeness centrality": dependency.closeness_centrality,
            "outdegree centralization": dependency.outdegree_centralization,
            "closeness centralization": dependency.closeness_centralization,
            "longest shortest path": dependency.longest_shortest_path,
            "dependents per word": dependency.dependents_per_word,
        }

    def _read_text(self, text: str):

        try:
            # Transform the text into conll-u format
            doc = self.nlp(text)

            CoNLL.write_doc2conll(doc, "temp.conllu")

            with open("temp.conllu", "r") as f:
                sentences, graphs = zip(*conllu.read_conllu_sentences(f, ignore_case=self.ignore_case))
                tokens = list(itertools.chain.from_iterable(sentences))

            os.remove("temp.conllu")

            return sentences, graphs, tokens
        except Exception as e:
            logger.warning(f"{e}. The text '{text}' could not be processed!")
            return None, None, None

    def surfaced_based_metrics(self, text: str, metrics: list[str] = None):

        try:
            _, _, tokens = self._read_text(text)

            if metrics is None:
                metrics = list(self._surfaced_based_metrics.keys())

            # If the text could not be processed
            if tokens is None:
                return {metric_name: None for metric_name in metrics}

            results = {}
            for (name, metric_method) in self._surfaced_based_metrics.items():
                if name in metrics:
                    mean, stdev, res = misc.bootstrap(metric_method, tokens, len(tokens), strategy="spread")
                    results[name] = mean

            return results
        except Exception as e:
            logger.warning(f"{e}. The text '{text}' could not be processed!")
            return {metric_name: None for metric_name in metrics}

    def pos_based_metrics(self, text: str, metrics: list[str] = None):

        try:
            _, _, tokens = self._read_text(text)

            if metrics is None:
                metrics = list(self._pos_based_metrics.keys())

            # If the text could not be processed
            if tokens is None:
                return {metric_name: None for metric_name in metrics}

            text = Text.from_tokens(tokens)

            results = {}
            for name, metric_method in self._pos_based_metrics.items():
                if name in metrics:
                    results[name] = metric_method(text)

            return results
        except Exception as e:
            logger.warning(f"{e}. The text '{text}' could not be processed!")
            return {metric_name: None for metric_name in metrics}

    def dependency_based_metrics(self, text: str, metrics: list[str] = None):

        try:
            _, graphs, _ = self._read_text(text)

            if metrics is None:
                metrics = list(self._dependency_based_metrics.keys())

            if graphs is None:
                return {metric_name: None for metric_name in metrics}

            results = {}
            for name, metric_method in self._dependency_based_metrics.items():
                if name in metrics:
                    value, stdev = metric_method(graphs)
                    results[name] = value

            return results
        except Exception as e:
            logger.warning(f"{e}. The text '{text}' could not be processed!")
            return {metric_name: None for metric_name in metrics}

    def complexity(self, text: str) -> dict:

        results = {}

        _, graphs, tokens = self._read_text(text)

        results = self.surfaced_based_metrics(text, metrics=["type-token ratio", "evenness"])
        results.update(self.pos_based_metrics(text, metrics=["lexical_density", "rarity"]))

        results.update(self.dependency_based_metrics(text, metrics=["average dependency distance",
                                                                    "dependents per word"]))

        return results

"""An interface for executing programs."""

from allennlp.common.registrable import Registrable

from fertility.eval.geo_eval.executor_geo import ProgramExecutorGeo
from fertility.eval.lev import MyMetric
from fertility.tree_utils.geo_utils import geo_reconstruct_tree
from fertility.tree_utils.read_funql import tree2funql


class Executor(Registrable):

    def execute(self, program: str, kb_str: str = None) -> str:
        raise NotImplementedError


import gc
from typing import List, Dict
from overrides import overrides
from allennlp.training.metrics import Metric


@MyMetric.register("geo_acc")
class GeoDenotationAccuracy(MyMetric):
    """
    Denotation accuracy based on program executions.
    """

    def __init__(self, insert_quotes: bool = True) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.
        self._executor = ProgramExecutorGeo()
        self.insert_quotes = insert_quotes

    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    def prepare(self, pred: List[str]):
        try:
            t = geo_reconstruct_tree(pred, self.insert_quotes)
            return tree2funql(t)
        except (IndexError, AssertionError):
            return "failed!"

    @overrides
    def add_instances(self, predictions: List[List[str]], gold_targets: List[List[str]]) -> None:
        self._total_counts += len(predictions)
        if self._total_counts % 1000 == 0:  # collect garbage once in a while
            gc.collect()

        for i, predicted_tokens in enumerate(predictions):

            gold_tokens = gold_targets[i] if gold_targets is not None else ['no_targets']

            denotation = self._executor.execute(self.prepare(predicted_tokens))

            gold_answer = self._executor.execute(self.prepare(gold_tokens))

            if gold_answer.startswith("error_parse:"):
                print("Warning, problem with denotation of gold program:", gold_tokens)

            if gold_answer == denotation:
                self._correct_counts += 1

                # if predicted_tokens != gold_tokens:
                #     print("prediction", predicted_tokens, self.prepare(predicted_tokens))
                #     print("gold", gold_tokens, self.prepare(gold_tokens))
                #     print(denotation)
                #     print("===")


    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_counts == 0:
            accuracy = 0.
        else:
            accuracy = self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {"geo_acc": accuracy}

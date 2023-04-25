from collections import defaultdict
from typing import Dict, List

from fertility.eval.lev import MyMetric


@MyMetric.register("acc_by_length")
class AccByLength(MyMetric):

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_instances = defaultdict(int)
        self.correct_instances = defaultdict(int)

    def get_metric(self, reset: bool) -> Dict[str, float]:
        d = {"seq_acc_"+str(k): self.correct_instances[k]/self.total_instances[k] if self.total_instances[k] > 0 else 0.0
                                                        for k in self.total_instances}

        if reset:
            self.reset()
        return d

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)

        for p, g in zip(predictions, gold):
            self.total_instances[len(g)] += 1
            self.correct_instances[len(g)] += (p == g)


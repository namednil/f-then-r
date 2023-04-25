import logging
from typing import Dict, List


logger = logging.getLogger(__name__)
class NLGMetrics:

    nlg_eval = None

    def __init__(self):
        from nlgeval import NLGEval
        self.reset()
        if self.nlg_eval is None:
            logger.info("Loading NLGEval...")
            self.nlg_eval = NLGEval()
            logger.info("done.")

    def reset(self):
        self.all_preds = []
        self.all_gold = []

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if not reset or len(self.all_gold) == 0:
            return dict()

        assert len(self.all_preds) == len(self.all_gold)
        m = self.nlg_eval.compute_metrics([self.all_gold], self.all_preds) # there is only a single reference text.
        self.reset()
        return {k: float(v) for k, v in m.items()}

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        self.all_preds.extend([" ".join(p) for p in predictions])
        # self.all_gold.extend([" ".join(g) for g in gold])
        self.all_gold.extend(gold)

        assert len(self.all_preds) == len(self.all_gold)


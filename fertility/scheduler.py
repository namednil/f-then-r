from allennlp.common import Registrable


class RateScheduler(Registrable):

    def get_rate(self, epoch: int) -> float:
        raise NotImplementedError()


@RateScheduler.register("on_off")
class OnOffRateScheduler(RateScheduler):

    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs

    def get_rate(self, epoch: int) -> float:
        return 0.0 if epoch < self.num_epochs else 1.0

@RateScheduler.register("constant")
class OnOffRateScheduler(RateScheduler):

    def __init__(self, constant: float):
        self.constant = constant

    def get_rate(self, epoch: int) -> float:
        return self.constant


@RateScheduler.register("linear")
class LinearScheduler(RateScheduler):

    def __init__(self, begin_epoch: int, num_epochs: int, max_value: float):
        self.begin_epoch = begin_epoch
        self.total_epochs = num_epochs
        self.max_value = max_value

    def get_rate(self, epoch: int) -> float:
        if epoch < self.begin_epoch:
            return 0.0
        rel_epoch = epoch - self.begin_epoch
        d = self.max_value / (self.total_epochs - self.begin_epoch)
        return rel_epoch * d
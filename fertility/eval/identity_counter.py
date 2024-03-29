import torch


class EntropyCounter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.instances = 0
        self.entropy = 0.0

    def add_matrix(self, batched_matrix: torch.Tensor, target_mask: torch.Tensor):
        bm = batched_matrix + 1e-12
        self.instances += batched_matrix.shape[0]
        entropy = -(bm * torch.log(bm)).sum(dim=[1, 2]) / target_mask.sum(dim=-1, keepdim=True)
        self.entropy += entropy.detach().sum().cpu().numpy()

    def get_metrics(self, reset):
        if self.instances == 0:
            return {"perm_entropy" : 0}
        a = self.entropy / self.instances
        if reset:
            self.reset()
        return {"perm_entopy": a}


class IdentityCounter:

    def __init__(self, thresh):
        self.reset()
        self.thresh = thresh

    def reset(self):
        self.instances = 0
        self.identities = 0

    def add_matrix(self, batched_matrix: torch.Tensor, target_lengths: torch.Tensor):
        """

        :param batched_matrix: shape (batch_size, n, n)
        :return:
        """
        n = batched_matrix.shape[1]
        assert batched_matrix.shape[2] == n
        d = torch.eye(n, device=batched_matrix.device).unsqueeze(0) * batched_matrix
        s = d.sum(dim=[1, 2]) #shape (batch_size,)
        self.instances += s.shape[0]
        self.identities += ((s - target_lengths)**2 < self.thresh).sum().cpu().numpy()

    def get_metrics(self, reset: bool):
        if self.instances == 0:
            return {"frac_identities": 0}
        frac = self.identities / self.instances
        if reset:
            self.reset()
        return {"frac_identities": frac}
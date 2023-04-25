import numba
import torch
import numpy as np

@numba.jit(nopython=True)
def compute_joint_numba(forward_sum, backward_sum, px, target_lengths, J):
    # Assumes that forward_sum and backward_sum have been padded already.
    batch_size, n, max_l = forward_sum.shape
    l = max_l - 1
    m = px.shape[2]
    n -= 1  # subtract the padding that we assumed is there, we pad forward_sum and backward_sum with 1 each
    for b in range(batch_size):
        bl = int(target_lengths[b])
        for i in range(n):
            for j in range(bl):
                for r1 in range(1, min(m, j+1+1)):
                    for r2 in range(min(m-r1, bl-j)):
                        J[b, i, j, r1] += forward_sum[b, i, j+1-r1] * px[b, i, r1+r2] * backward_sum[b, i+1, bl - (j+1+r2)]
                        # not until i-1 but until forward_sum[b,i, ...] because we added a vector to the beginning.


@numba.jit(nopython=True)
def compute_joint_grad(forward_sum, backward_sum, px, target_lengths, grad_j, d_forward, d_backward, d_px):
    batch_size, n, max_l = forward_sum.shape
    l = max_l - 1
    m = px.shape[2]
    n -= 1
    for b in range(batch_size):
        bl = int(target_lengths[b])
        for i in range(n):
            for j in range(bl):
                for r1 in range(1, min(m, j+1+1)):
                    for r2 in range(min(m-r1, bl-j)):
                        # J[b, i, j, r1] += forward_sum[b, i, j+1-r1] * px[b, i, r1+r2] * backward_sum[b, i+1, bl - (j+1+r2)]
                        # not until i-1 but until forward_sum[b,i, ...] because we added a vector to the beginning.
                        d_px[b, i, r1+r2] += grad_j[b, i, j, r1] * forward_sum[b, i, j+1-r1] * backward_sum[b, i+1, bl - (j+1+r2)]
                        d_forward[b, i, j+1-r1] += grad_j[b, i, j, r1] * px[b, i, r1+r2] * backward_sum[b, i+1, bl - (j+1+r2)]
                        d_backward[b, i+1, bl - (j+1+r2)] += grad_j[b, i, j, r1] * forward_sum[b, i, j+1-r1] * px[b, i, r1+r2]




class FertilityLayerJoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, forward_sum, backward_sum, px, target_lengths):
        f, b, px_n, tgt_l = forward_sum.detach().cpu().numpy(), backward_sum.detach().cpu().numpy(), px.detach().cpu().numpy(), target_lengths.cpu().numpy()
        batch_size, n, max_l = forward_sum.shape
        # Concatenate a vector to the beginning of forward_sum and a vector to the end of backward_sum
        # to handle the boundary.
        a = np.zeros((batch_size, 1, max_l), dtype=f.dtype)
        a[:, 0, 0] = 1.0  # probability is one that this sums to 0 (because there is nothing there!)
        f = np.concatenate([a, f], axis=1)

        a = np.zeros((batch_size, 1, max_l), dtype=f.dtype)
        a[:, -1, 0] = 1.0  # probability of 1.0 that sum coming from the right is 0 (because there is nothing there)
        b = np.concatenate([b, a], axis=1)

        ctx.saved = f,b,px_n, tgt_l
        ctx.device = forward_sum.device

        l = max_l - 1
        J = np.zeros((batch_size, n, l, px.shape[2]), dtype=f.dtype)
        # J = np.zeros((batch_size, n, l, px.shape[2])) #important for grad check to use higher precision!
        compute_joint_numba(f, b, px_n, tgt_l, J)
        return torch.from_numpy(J).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_j):
        f,b,px_n, tgt_l = ctx.saved
        d_forward = np.zeros(f.shape, dtype=f.dtype)
        d_backward = np.zeros(b.shape, dtype=b.dtype)
        d_px = np.zeros(px_n.shape, dtype=px_n.dtype)
        # d_px = np.zeros(px_n.shape)
        compute_joint_grad(f, b, px_n, tgt_l, grad_j.cpu().numpy(), d_forward, d_backward, d_px)
        # Remove the padding that we introduced:
        d_forward = d_forward[:, 1:, :]
        d_backward = d_backward[:, :-1, :]

        return torch.from_numpy(d_forward).to(ctx.device), torch.from_numpy(d_backward).to(ctx.device), torch.from_numpy(d_px).to(ctx.device), None #no gradient for target_lengths



def fertility_layer_joint(forward_sum, backward_sum, px, target_lengths, normalize: bool = True):
    batch_size = forward_sum.shape[0]
    n = forward_sum.shape[1]
    simple_fertility_result_joint = FertilityLayerJoint.apply(forward_sum, backward_sum, px, target_lengths)

    if normalize:
        # The custom implementation doesn't do a final division by the probability of respective target length.
        # so we do this here:
        divide_by = (backward_sum[torch.arange(batch_size, device=forward_sum.device), 0, target_lengths]).unsqueeze(1).unsqueeze(2).unsqueeze(3) #shape (batch_size, 1, 1, 1)
        simple_fertility_result_joint = simple_fertility_result_joint / divide_by
    return simple_fertility_result_joint



if __name__ == "__main__":
    # cd to this directory and run with "python fertility_numba.py"
    from conv_utils import cumulative_sum
    torch.manual_seed(48)
    # Create batch of two examples, with sequence length 4.
    # Each element in the input can take a fertility between 0 and 2, and we generate a distribution:
    b, isl, m = 2,4,3
    fertilities = torch.softmax(torch.randn((b,isl,m), requires_grad=True), dim=-1)

    # The first element in the batch will have output length 4, the second one output length 5:
    target_lengths = torch.tensor([4, 5])
    max_l = (torch.max(target_lengths).numpy()+1)*m

    #Compute forward probabilities:
    forward_sum = cumulative_sum(fertilities,
                                 max_l)  # shape (batch_size, input_seq_len, max_l) where forward_sum[b, i, n] is P(F_0 + ... F_i = n) for that batch element.
    backward_sum = torch.flip(cumulative_sum(torch.flip(fertilities, [1]), max_l), [1])

    # Compute
    fert = fertility_layer_joint(forward_sum, backward_sum, fertilities, target_lengths)
    # fert[b,i,j,k] is the marginal probability that in batch element b, input position i is mapped to output position
    # j AND that this is the k-th copy of i. (i.e. eq (9) in the paper)
    print(fert.shape)
    # show matrix with the information about copies marginalised out
    print(fert.sum(dim=-1))

    #Check that manual gradient implementation is correct:
    print("Grad check")
    torch.autograd.gradcheck(fertility_layer_joint, (forward_sum.double(), backward_sum.double(), fertilities.double(), target_lengths))

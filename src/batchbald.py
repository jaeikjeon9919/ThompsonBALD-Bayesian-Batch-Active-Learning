# AUTOGENERATED! DO NOT EDIT! File to edit: 01_batchbald.ipynb (unless otherwise specified).

__all__ = ['compute_conditional_entropy', 'compute_entropy', 'CandidateBatch', 'get_batchbald_batch', 'get_bald_batch']

# Cell
from dataclasses import dataclass
from typing import List
import torch
import math
from tqdm.auto import tqdm

from toma_1 import toma
import numpy as np

from src.BatchBALD import joint_entropy

# Cell

def compute_conditional_entropy(probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    # pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(probs_N_K_C, 1024)
    def compute(probs_n_K_C, start: int, end: int):
        nats_n_K_C = probs_n_K_C * torch.log(probs_n_K_C)
        nats_n_K_C[probs_n_K_C ==0] = 0.

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        # pbar.update(end - start)

    # pbar.close()

    return entropies_N


def compute_entropy(probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    # pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(probs_N_K_C, 1024)
    def compute(probs_n_K_C, start: int, end: int):
        mean_probs_n_C = probs_n_K_C.mean(dim=1)
        nats_n_C = mean_probs_n_C * torch.log(mean_probs_n_C)
        nats_n_C[mean_probs_n_C ==0] = 0.

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        # pbar.update(end - start)

    # pbar.close()

    return entropies_N

# Internal Cell
# Not publishing these at the moment.

def compute_conditional_entropy_from_logits(logits_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = logits_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    # pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(logits_N_K_C, 1024)
    def compute(logits_n_K_C, start: int, end: int):
        nats_n_K_C = logits_n_K_C * torch.exp(logits_n_K_C)

        entropies_N[start:end].copy_(
            -torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        # pbar.update(end - start)

    # pbar.close()

    return entropies_N


def compute_entropy_from_logits(logits_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = logits_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    # pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(logits_N_K_C, 1024)
    def compute(logits_n_K_C, start: int, end: int):
        mean_logits_n_C = torch.logsumexp(logits_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_logits_n_C * torch.exp(mean_logits_n_C)

        entropies_N[start:end].copy_(
            -torch.sum(nats_n_C, dim=1))
        # pbar.update(end - start)

    # pbar.close()

    return entropies_N

# Cell


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def get_batchbald_batch(probs_N_K_C: torch.Tensor,
                        batch_size: int,
                        num_samples: int,
                        dtype=None,
                        device=None) -> CandidateBatch:
    N, K, C = probs_N_K_C.shape
    print("N, K, C", N, K, C)
    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(probs_N_K_C) # shape: pool_size

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(num_samples,
                                                            batch_size - 1,
                                                            K,
                                                            C,
                                                            dtype=dtype,
                                                            device=device)

    # We always keep these on the CPU.

    scores_N = torch.zeros(N, dtype=torch.double, pin_memory=torch.cuda.is_available()) # shape: pool_size

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(
                probs_N_K_C[latest_index:latest_index + 1])
                # probs_N_K_C[latest_index:latest_index + 1] only store y_s in candidate indices
                # this return # \hat P_{1:n-1} (c^{n-1} x k)

        shared_conditinal_entropies = conditional_entropies_N[
            candidate_indices].sum()

        batch_joint_entropy.compute_batch(probs_N_K_C,
                                          output_entropies_B=scores_N)
        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float('inf')

        candidate_score, candidate_index = scores_N.max(dim=0)
        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())
    return CandidateBatch(candidate_scores, candidate_indices)

# Cell


def get_bald_batch(probs_N_K_C: torch.Tensor,
                   batch_size: int,
                   dtype=None,
                   device=None) -> CandidateBatch:
    N, K, C = probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(probs_N_K_C)
    scores_N += compute_entropy(probs_N_K_C)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist()), scores_N


##
def get_bald(probs_N_K_C):
    N, K, C = probs_N_K_C.shape




    scores_N = -compute_conditional_entropy(probs_N_K_C)
    scores_N += compute_entropy(probs_N_K_C)
    # print("gamma = 0.7")

    return scores_N


##
def comp_cond_entropy(probs_N_K_C):
    N, K, C = probs_N_K_C.shape
    probs_n_K_C = probs_N_K_C * torch.log(probs_N_K_C)
    probs_n_K_C[probs_n_K_C == 0] = 0.
    return -torch.sum(probs_n_K_C, dim=(2))



##
def compute_var_ratio(probs_N_K_C):
    # probs_N_K_C[probs_N_K_C == 0] = 0.

    mean_probs_n_C = probs_N_K_C.mean(dim=1)
    max_values, max_indicies = (mean_probs_n_C).max(axis=1)
    variation_ratios = 1-max_values
    return variation_ratios



##
def get_batchbald_two_batch(probs_N_K_C: torch.Tensor,
                        batch_size: int,
                        num_samples: int,
                        num_sample_Inner_BatchBALD: int,
                        dtype=None,
                        device=None) -> CandidateBatch:
    N, K, C = probs_N_K_C.shape
    print("N, K, C", N, K, C)
    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    candidate_indices_temp = []
    candidate_scores_temp = []


    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)
    variation_ratios_idx = compute_var_ratio(probs_N_K_C)
    candiate_scores, candidate_indices = torch.topk(variation_ratios_idx, int(batch_size/num_sample_Inner_BatchBALD))

    candiate_scores = candiate_scores.cpu().numpy()
    candidate_indices = candidate_indices.cpu().numpy()
    candidate_indices = candidate_indices.astype(int)
    conditional_entropies_N = compute_conditional_entropy(probs_N_K_C) # shape: pool_size


    candidate_indices_for_loop = candidate_indices.copy()
    # batch_joint_entropy = joint_entropy.DynamicJointEntropy(num_samples,
    #                                                         batch_size - 1,
    #                                                         K,
    #                                                         C,
    #                                                         dtype=dtype,
    #                                                         device=device)

    # We always keep these on the CPU.

    # scores_N = torch.zeros(N, dtype=torch.double, pin_memory=torch.cuda.is_available()) # shape: pool_size
    for i in tqdm(candidate_indices_for_loop.astype(int), desc="BatchBALD", leave=False):
        # print("")
        batch_joint_entropy = joint_entropy.DynamicJointEntropy(num_samples,
                                                                batch_size - 1,
                                                                K,
                                                                C,
                                                                dtype=dtype,
                                                                device=device)
                                                                
        candidate_indices_temp = []
        candidate_scores_temp = []
        scores_N = torch.zeros(N, dtype=torch.double, pin_memory=torch.cuda.is_available()) # shape: pool_size
        # probs_N_K_C[candidate_indices] = 1e-12
        # candidate_indices_temp = np.hstack((candidate_indices_temp, i))
        # candidate_indices_temp = candidate_indices_temp.astype(int)

        for j in range(num_sample_Inner_BatchBALD):
            if j == 0:
                shared_conditinal_entropies = conditional_entropies_N[
                    candidate_indices_temp].sum()
                # print("")
                # print("conditional_entropies_N[candidate_indices_temp]", conditional_entropies_N[candidate_indices_temp])
                # print("shared_conditinal_entropies shape", shared_conditinal_entropies.shape)


                # print("1 scores_N: ", scores_N)
                # print("scores_N1:", scores_N)
                batch_joint_entropy.compute_batch(probs_N_K_C,
                                                output_entropies_B=scores_N)
                # print("scores_N2:", scores_N)
                # print("2 scores_N: ", scores_N)
                
                scores_N -= conditional_entropies_N + shared_conditinal_entropies

                # print("3 scores_N: ", np.array(scores_N).tolist())


                # print("conditional_entropies_N", conditional_entropies_N)
                # print("conditional_entropies_N shape", conditional_entropies_N.shape)

                # print("4 scores_N: ", scores_N)

                scores_N[candidate_indices_temp] = -float('inf')

                # candidate_score, candidate_index = scores_N.max(dim=0)
                candidate_indices_temp = np.hstack((candidate_indices_temp, i))
                candidate_scores_temp = np.hstack((candidate_scores_temp, scores_N[i]))
                # print("candidate_indices chosen from VarRatio: ", candidate_indices_temp)
                # print("candidate_scoress chosen from VarRatio: ", candidate_scores_temp)
                scores_N[candidate_indices] = -float('inf')


            else:
                if j > 0:
                    # print("candidate_indices_temp", candidate_indices_temp)
                    latest_index = int(candidate_indices_temp[-1])
                    batch_joint_entropy.add_variables(
                        probs_N_K_C[latest_index:latest_index + 1])
                        # probs_N_K_C[latest_index:latest_index + 1] only store y_s in candidate indices
                        # this return # \hat P_{1:n-1} (c^{n-1} x k)
                shared_conditinal_entropies = conditional_entropies_N[
                    candidate_indices_temp].sum()
                # print("")
                # print("conditional_entropies_N[candidate_indices_temp]", conditional_entropies_N[candidate_indices_temp])
                # print("shared_conditinal_entropies shape", shared_conditinal_entropies.shape)


                # print("1 scores_N: ", scores_N)
                # print("scores_N1:", scores_N)
                batch_joint_entropy.compute_batch(probs_N_K_C,
                                                output_entropies_B=scores_N)
                # print("scores_N2:", scores_N)
                # print("2 scores_N: ", scores_N)
                
                scores_N -= conditional_entropies_N + shared_conditinal_entropies

                # print("3 scores_N: ", scores_N)


                # print("conditional_entropies_N", conditional_entropies_N)
                # print("conditional_entropies_N shape", conditional_entropies_N.shape)

                scores_N[candidate_indices_temp] = -float('inf')
                scores_N[candidate_indices] = -float('inf')
                candidate_score, candidate_index = scores_N.max(dim=0)
                candidate_indices_temp = np.hstack((candidate_indices_temp, candidate_index.item()))
                candidate_scores_temp = np.hstack((candidate_scores_temp, candidate_score.item()))
                candidate_indices_temp = candidate_indices_temp.astype(int)
        # print("candidate_indices chosen from BatchBALD: ", candidate_indices_temp.tolist())
        # print("candidate_scores chosen from BatchBALD: ", candidate_scores_temp.tolist())
            # scores_N[candidate_indices_temp] = -float('inf')
        # print("last candidate_indices_temp: ", candidate_indices_temp)
        probs_N_K_C[candidate_indices_temp] = 1e-18
        # probs_N_K_C[candidate_indices_temp[-1]] = 1e-18
        candidate_indices = np.hstack((candidate_indices, candidate_indices_temp[1:]))
        candidate_scores = np.hstack((candidate_scores, candidate_scores_temp))
    print("candidate_indices:", candidate_indices.tolist())
    print("candidate_scores:", candidate_scores.tolist())
    # print("len(candidate_indices)", len(candidate_indices))
    return CandidateBatch(candidate_scores, candidate_indices.astype("int"))



def alpha_divergence(probs_N_K_C, batch_size):
    alpha = 2
    N, K, C = probs_N_K_C.shape

    p_y_alpha_m_1 = probs_N_K_C.mean(dim=1) ** (alpha - 1)

    p_y_given_w_alpha = probs_N_K_C ** alpha
    nats_p_y_given_w_alpha = torch.sum(p_y_given_w_alpha, dim=(1)) / K

    scores_N = 1 / (1 - alpha) * torch.log((p_y_alpha_m_1 * nats_p_y_given_w_alpha).sum(dim=1))

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist()), scores_N


def thommpson_bald(probs_N_K_C, batch_size, sample_size=300):
    N, K, C = probs_N_K_C.shape

    mean_prob_N_C = probs_N_K_C.mean(dim=1)
    nats_N_C = mean_prob_N_C * torch.log(mean_prob_N_C)
    entropy_N = -torch.sum(nats_N_C, dim=1)
    perm = torch.tensor((torch.randperm(sample_size).numpy().tolist() * 50))
    idx = perm[:batch_size]
    candidate_score = []
    candidate_idx = []
    for i in idx:
        sample_cond_entropy_N = -torch.sum(probs_N_K_C[:,i,:] * torch.log(probs_N_K_C[:,i,:]), dim=1)
        thompson_bald = entropy_N - sample_cond_entropy_N
        thompson_bald[candidate_idx] = -float('inf') # 0
        score, idx = torch.topk(thompson_bald, 1)
        candidate_score.append(score.cpu().numpy().squeeze().tolist())
        candidate_idx.append(idx.cpu().numpy().squeeze().tolist())
    bald_score = get_bald(probs_N_K_C)
    return CandidateBatch(candidate_score, candidate_idx), bald_score, bald_score[candidate_idx].numpy().tolist()


def thommpson_likelihood_bald(probs_N_K_C, batch_size, sample_size=1000):
    N, K, C = probs_N_K_C.shape

    mean_prob_N_C = probs_N_K_C.mean(dim=1)
    nats_N_C = mean_prob_N_C * torch.log(mean_prob_N_C)
    entropy_N = -torch.sum(nats_N_C, dim=1)
    perm = torch.tensor((torch.randperm(sample_size).numpy().tolist() * 50))
    idx = perm[:batch_size]
    candidate_score = []
    candidate_idx = []
    for i in idx:
        sample_cond_entropy_N = -torch.sum(mean_prob_N_C * torch.log(probs_N_K_C[:,i,:]), dim=1)
        thompson_bald = entropy_N + sample_cond_entropy_N
        thompson_bald[candidate_idx] = 0
        score, idx = torch.topk(thompson_bald, 1)
        candidate_score.append(score.cpu().numpy().squeeze().tolist())
        candidate_idx.append(idx.cpu().numpy().squeeze().tolist())
    bald_score = get_bald(probs_N_K_C)
    return CandidateBatch(candidate_score, candidate_idx), bald_score



def FULL_thommpson_bald(probs_N_K_C, batch_size, sample_size=300):
    N, K, C = probs_N_K_C.shape
    mean_prob_N_C = probs_N_K_C.mean(dim=1)
    thompson_samples = (probs_N_K_C * torch.log(probs_N_K_C / mean_prob_N_C.unsqueeze_(-1).expand(N, C, K).permute(0, 2, 1))).sum(2)

    perm = torch.tensor((torch.randperm(sample_size).numpy().tolist() * 50))
    idx = perm[:batch_size]
    candidate_score = []
    candidate_idx = []
    for i in idx:
        thompson_bald = thompson_samples[:,i]
        thompson_bald[candidate_idx] = -float('inf')
        score, idx = torch.topk(thompson_bald, 1)
        candidate_score.append(score.cpu().numpy().squeeze().tolist())
        candidate_idx.append(idx.cpu().numpy().squeeze().tolist())
    bald_score = get_bald(probs_N_K_C)
    return CandidateBatch(candidate_score, candidate_idx), bald_score, bald_score[candidate_idx].numpy().tolist()


def Boltzmann_BALD(probs_N_K_C, batch_size, temperature):
    bald_score = get_bald(probs_N_K_C)

    bald_prob = torch.exp(bald_score / temperature) / torch.exp(bald_score / temperature).sum()

    candidate_idx = torch.multinomial(bald_prob, batch_size, replacement=False)
    return CandidateBatch(bald_score[candidate_idx].numpy().tolist(), candidate_idx)



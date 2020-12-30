##
import torch
import math

import os

import argparse


from torch.nn import functional as F
import torch.utils.data as data
from tqdm.auto import tqdm

import numpy as np
from src.BatchBALD import active_learning, repeated_mnist, nn_model
# from src.BatchBALD.utils import EarlyStopping
import matplotlib.pyplot as plt

from src.BatchBALD import batchbald

from archive.active_bayesian_coresets.acs import ProjectedFrankWolfe
import src.Bayesian_coresets.acs.utils as utils

import pickle



parser = argparse.ArgumentParser()


parser.add_argument("--save_dir", default='./result/MC-DropOut/BatchBALD', dest = "save_dir", help="Save directory")
parser.add_argument("--result_file_name_index", default='1', dest = "result_file_name_index", help="Result File Name")
parser.add_argument("--num_initial_samples", type=int, default=100, dest = "num_initial_samples", help="Number of labeled observations in dataset")
parser.add_argument("--dataset", default='digits_mnist', dest = "dataset", help="Torchvision dataset")

# optimization params
parser.add_argument('--n_epochs', type=int, default=150, dest = "n_epochs", help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=64, dest = "batch_size", help='batch_size')
parser.add_argument('--test_batch_size', type=int, default=512, dest = "test_batch_size", help='test_batch_size')
parser.add_argument('--scoring_batch_size', type=int, default=128, dest = "scoring_batch_size", help='scoring_batch_size')
parser.add_argument('--num_test_inference_samples', type=int, default=10, dest = "num_test_inference_samples", help='number of MC samples we will use in test phase')
parser.add_argument('--num_inference_samples', type=int, default=100, dest = "num_inference_samples", help='number of MC samples for inference')
parser.add_argument('--num_samples', type=int, default=10000, dest = "num_samples", help='num_samples')


# active learning params
parser.add_argument('--max_training_samples', type=int, default=1000, dest = "max_training_samples", help='Active learning budget')
parser.add_argument('--acquisition_batch_size', type=int, default=50, dest = "acquisition_batch_size", help='Active learning batch size')
parser.add_argument('--acq', default='BatchBALD', dest = "acq", help='Active learning acquisition function (Sparse Subset Approximation, BatchBALD, BALD')
parser.add_argument('--number_of_remove_data', type=int, default=0, dest = "number_of_remove_data", help='This removes most of the pool data')
parser.add_argument('--num_repeat', type=int, default=1, dest = "num_repeat", help='number of dataset repetition')
parser.add_argument('--num_projections', type=int, default=1, dest = "num_projections", help='number of projections')
parser.add_argument('--temperature', type=float, default=1, dest = "temperature", help='Boltzmann temperature')






args = parser.parse_args()


save_dir = args.save_dir
result_file_name_index = args.result_file_name_index
num_initial_samples = args.num_initial_samples
dataset = args.dataset

n_epochs = args.n_epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size
scoring_batch_size = args.scoring_batch_size
num_test_inference_samples = args.num_test_inference_samples
num_inference_samples = args.num_inference_samples
num_samples = args.num_samples


max_training_samples = args.max_training_samples
acquisition_batch_size = args.acquisition_batch_size
acq = args.acq
number_of_remove_data = args.number_of_remove_data

num_repeat = args.num_repeat


num_projections = args.num_projections
temperature = args.temperature


##
train_dataset, test_dataset = repeated_mnist.create_repeated_MNIST_dataset(num_repetitions=num_repeat, add_noise=False, dataset=dataset)

## select indices of initial samples
num_classes = 10

initial_samples = active_learning.get_balanced_sample_indices(
    repeated_mnist.get_targets(train_dataset),
    num_classes=num_classes,
    n_per_digit=num_initial_samples/num_classes)


if acq not in ["BatchBALD", "BALD", "Random", "Thompson_BALD", "ACS-FW", "FULL_Thompson_BALD", "Boltzmann_BALD"]:
    raise ValueError('Invalid inference method: {}'.format(acq))







use_cuda = torch.cuda.is_available()

print(f"use_cuda: {use_cuda}")

device = "cuda" if use_cuda else "cpu"

kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


##

active_learning_data = active_learning.ActiveLearningData(train_dataset)



## take initial indices and update the training set
active_learning_data.acquire(initial_samples)


## this removes most of the pool data
if number_of_remove_data > 0:
    active_learning_data.extract_dataset_from_pool(number_of_remove_data)


##
train_loader = torch.utils.data.DataLoader(
    active_learning_data.training_dataset,
    shuffle = False,
    batch_size=batch_size,
    **kwargs,
)



pool_loader = torch.utils.data.DataLoader(active_learning_data.pool_dataset,
                                          batch_size=scoring_batch_size,
                                          shuffle=False,
                                          **kwargs)


test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=False,
                                          **kwargs)



title_str = '{} {} (Initial # of Labeled={}, Batch size={}, budget={}, num_repeat={}, num_samples={}, num_inference_samples={})'.format(acq, args.dataset, num_initial_samples, acquisition_batch_size, max_training_samples, num_repeat, num_samples,num_inference_samples)


print((save_dir, title_str + "_" + result_file_name_index + '.pkl'))


## Run experiment

print('==============================================================================================')
print(title_str)
print('==============================================================================================')


print("")


test_accs = []
test_loss = []
added_indices = []

pbar = tqdm(initial=len(active_learning_data.training_dataset),
            total=max_training_samples,
            desc="Training Set Size")
test_performances = {'Acc': [],  'batch': []}



while True:
    model = nn_model.BayesianCNN(num_classes).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())




    for epoch in range(1, n_epochs+1):
        model.train()

        # Train
        for data, target in train_loader:
            data = data.to(device=device)
            target = target.to(device=device)

            optimizer.zero_grad()
            prediction = model(data, 1).squeeze(1)    # model(data, # of MC samples)
            loss = F.nll_loss(prediction, target)

            loss.backward()
            optimizer.step()



    # Test
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device=device)
            target = target.to(device=device)

            prediction = torch.logsumexp(
                model(data, num_test_inference_samples),
                dim=1) - math.log(num_test_inference_samples)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)
    test_loss.append(loss)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)
    test_accs.append(percentage_correct)
    # if write:
    #     writer_test.add_scalar('test-acc', (percentage_correct), len(active_learning_data.training_dataset))

    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
        loss, correct, len(test_loader.dataset), percentage_correct))

    if len(active_learning_data.training_dataset) >= max_training_samples:
        break

    #============================#
    #====Active Learning step====#
    #============================#
    # Acquire pool predictions
    N = len(active_learning_data.pool_dataset)


    if acq != "ACS-FW":
        logits_N_K_C = torch.empty((N, num_inference_samples, num_classes),
                                   dtype=torch.double,
                                   pin_memory=use_cuda)
        with torch.no_grad():
            model.eval()

            for i, (data, _) in enumerate(pool_loader):
                data = data.to(device=device)
                lower = i * pool_loader.batch_size
                upper = min(lower + pool_loader.batch_size, N)
                logits_N_K_C[lower:upper].copy_(model(
                    data, num_inference_samples).double(),
                                                non_blocking=True)
    else:
        logits_N_K_C_temp = torch.empty((N, num_inference_samples + num_projections, num_classes),
                                        dtype=torch.double,
                                        pin_memory=use_cuda)
        logits_N_K_C = torch.empty((N, num_inference_samples, num_classes),
                                   dtype=torch.double,
                                   pin_memory=use_cuda)
        logits_project_N_K_C = torch.empty((N, num_projections, num_classes),
                                           dtype=torch.double,
                                           pin_memory=use_cuda)

        with torch.no_grad():
            model.eval()
            for i, (data, _) in enumerate(pool_loader):

                data = data.to(device=device)
                lower = i * pool_loader.batch_size
                upper = min(lower + pool_loader.batch_size, N)

                logits_N_K_C_temp[lower:upper].copy_(model(
                    data, num_inference_samples + num_projections).double(),
                                                     non_blocking=True)

        logits_N_K_C[:, :, :].copy_(logits_N_K_C_temp[:, :num_inference_samples, :],
                                    non_blocking=True)
        logits_project_N_K_C[:, :, :].copy_(logits_N_K_C_temp[:, num_inference_samples:, :],
                                            non_blocking=True)



    if acq == "BALD":
        with torch.no_grad():
            candidate_batch = batchbald.get_bald_batch(logits_N_K_C.exp_(),
                                                            acquisition_batch_size,
                                                            num_samples)
        targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(candidate_batch.indices)
        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        print("Labels: ", targets[candidate_batch.indices])
        test_performances['batch'].append(dataset_indices)



        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))
    elif acq == "BatchBALD":
        with torch.no_grad():
            candidate_batch = batchbald.get_batchbald_batch(logits_N_K_C.exp_(),
                                                            acquisition_batch_size,
                                                            num_samples,
                                                            dtype=torch.double,
                                                            device=device
                                                            )
        targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(
            candidate_batch.indices)
        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        print("Labels: ", targets[candidate_batch.indices])
        test_performances['batch'].append(dataset_indices)



        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))

    elif acq == "Thompson_BALD":
        with torch.no_grad():
            candidate_batch = batchbald.thommpson_bald(logits_N_K_C.exp_(),
                                                            acquisition_batch_size,
                                                            num_inference_samples
                                                            )
        targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(
            candidate_batch.indices)
        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        print("Labels: ", targets[candidate_batch.indices])
        test_performances['batch'].append(dataset_indices)



        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))

    elif acq == "Boltzmann_BALD":
        with torch.no_grad():
            candidate_batch = batchbald.thommpson_bald(logits_N_K_C.exp_(),
                                                            acquisition_batch_size,
                                                            temperature
                                                            )
        targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(
            candidate_batch.indices)
        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        print("Labels: ", targets[candidate_batch.indices])
        test_performances['batch'].append(dataset_indices)



        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))


    elif acq == "FULL_Thompson_BALD":
        with torch.no_grad():
            candidate_batch = batchbald.FULL_thommpson_bald(logits_N_K_C.exp_(),
                                                            acquisition_batch_size,
                                                            num_inference_samples
                                                            )
        targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(
            candidate_batch.indices)
        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        print("Labels: ", targets[candidate_batch.indices])
        test_performances['batch'].append(dataset_indices)



        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))



    elif acq == "ACS-FW":
        coreset = ProjectedFrankWolfe
        probs_N_K_C = logits_N_K_C.clone().exp_()
        py = probs_N_K_C.clone().mean(1)
        projections = []
        ent_N = -(probs_N_K_C.mean(1) * torch.log(probs_N_K_C.mean(1))).sum(
            1)  # test2ent(test2.exp_().mean(1).log()).entropy()
        ys = (torch.ones_like(probs_N_K_C[:, 0, :]).type(torch.LongTensor) * torch.arange(10)[None, :]).t()
        for i in range(num_projections):
            loglik = -torch.stack(
                [torch.nn.functional.nll_loss(logits_project_N_K_C[:, i, :], y, reduction='none') for y in ys]).t()
            expected_loglik = torch.sum(py * loglik, dim=-1, keepdim=True)

            projections.append(expected_loglik + ent_N[:, None])
        ELn = utils.to_gpu(torch.sqrt(1 / torch.FloatTensor([num_projections])) * torch.cat(projections, dim=1))
        ELn = ELn.float()
        cs = coreset(ELn, ent_N, num_projections)
        batch = cs.build(acquisition_batch_size)

        targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(batch)
        print("Dataset indices: ", dataset_indices)
        print("Labels: ", targets[batch])
        test_performances['batch'].append(dataset_indices)


        active_learning_data.acquire(batch)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))


    elif acq == "Random":
        random_indices = active_learning_data.get_random_pool_indices(acquisition_batch_size)

        targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(
            random_indices)

        print("Dataset indices: ", dataset_indices)
        print("Labels: ", targets[random_indices])    

        active_learning_data.acquire(random_indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))

        






test_performances['Acc'].append(test_accs)


with open(os.path.join(save_dir, title_str + "_" + result_file_name_index + '.pkl'), 'wb') as handle:
    pickle.dump({title_str: test_performances}, handle)




# if write:
#     writer_test.close()
# if store:
#     np.save(os.path.join(result_dir, 'test_acc_6.npy'), test_accs)
#     np.save(os.path.join(result_dir, 'added_indices_1.npy'), added_indices)



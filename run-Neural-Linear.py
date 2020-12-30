import numpy as np
import torch

import argparse
import pickle
import time
import os

import src.Bayesian_coresets.acs.utils as utils
from src.Bayesian_coresets.acs.coresets import ProjectedFrankWolfe
from src.Bayesian_coresets.acs.al_data_set import Dataset, ActiveLearningDataset as ALD
from src.Bayesian_coresets.acs.coresets import Argmax, Random, KCenter, KMedoids

from src.Bayesian_coresets.resnet.resnets import resnet18

from src import batchbald

from copy import deepcopy
from tqdm.auto import tqdm


from torch.utils.data import DataLoader




parser = argparse.ArgumentParser()


parser.add_argument("--save_dir", default='./result/Neural_Linear/Sparse_Subset_Approximation', dest = "save_dir", help="Save directory")
parser.add_argument("--result_file_name_index", default='1', dest = "result_file_name_index", help="Result File Name")
parser.add_argument("--data_dir", default='./data', dest = "data_dir", help="Data directory")
parser.add_argument("--seed", type=int, default=222, dest = "seed", help="Random seed for data generation")
parser.add_argument("--init_num_labeled", type=int, default=20, dest = "init_num_labeled", help="Number of labeled observations in dataset")
parser.add_argument("--dataset", default='digits_mnist', dest = "dataset", help="Torchvision dataset")
parser.add_argument("--model_file", default='./models/best.pth.tar', dest = "model_file", help="Model directory")
parser.add_argument("--num_repetitions", type=int, default=1, dest="num_repetitions", help="number of data repetition")

# optimization params
parser.add_argument('--training_epochs', type=int, default=250, dest = "training_epochs", help='Number of training iterations')
parser.add_argument('--initial_lr', type=float, default=1e-3, dest = "initial_lr", help='Learning rate for optimization')
parser.add_argument('--freq_summary', type=int, default=100, dest = "freq_summary", help='Print frequency during training')
parser.add_argument('--weight_decay', type=float, default=5e-4, dest = "weight_decay", help='Add weight decay for feature extractor')
parser.add_argument('--weight_decay_theta', type=float, default=5e-4, dest = "weight_decay_theta", help='Add weight decay for linear layer')
parser.add_argument("--inference", default='MF', dest = "inference", help='Inference method (MF, Full, MCDropout')
parser.add_argument("--cov_rank", type=int, default=2, dest = "cov_rank", help='Rank of cov matrix for VI w/ full cov')

# active learning params
parser.add_argument('--threshold', type=float, default=1e12, dest = "threshold", help='threshold')
parser.add_argument('--budget', type=int, default=1000, dest = "budget", help='Active learning budget')
parser.add_argument('--batch_size', type=int, default=50, dest = "batch_size", help='Active learning batch size')
parser.add_argument('--acq', default='Sparse Subset Approximation', dest = "acq", help='Active learning acquisition function (Sparse Subset Approximation, BatchBALD, BALD')
parser.add_argument('--coreset', default='FW', dest = "coreset", help='Coreset construction (FW, GIGA)')
parser.add_argument('--num_projections', type=int, default=100, dest = "num_projections", help='Number of projections for acq=Proj')
parser.add_argument('--num_features', type=int, default=256, dest = "num_features", help='Number of features in feature extractor.')
parser.add_argument('--gamma', type=float, default=0., dest = "gamma", help='Parameter to trade off entropy term in projections')
parser.add_argument('--num_samples_IS', type=int, default=10000, dest = "num_samples_IS", help='num_samples_IS')
parser.add_argument('--num_sample_Inner_BatchBALD', type=int, default=10, dest = "num_sample_Inner_BatchBALD", help='num_sample_Inner_BatchBALD')
parser.add_argument('--num_inference_samples', type=int, default=100, dest = "num_inference_samples", help='num_inference_samples')
parser.add_argument('--temperature', type=float, default=0.1, dest = "temperature", help='temperature')



args = parser.parse_args()


## experiment
save_dir = args.save_dir
result_file_name_index = args.result_file_name_index
data_dir = args.data_dir
seed = args.seed
init_num_labeled = args.init_num_labeled
dataset = args.dataset
model_file = args.model_file
num_repetitions = args.num_repetitions

## optimization params
training_epochs = args.training_epochs
initial_lr = args.initial_lr
freq_summary = args.freq_summary
weight_decay = args.weight_decay
weight_decay_theta = args.weight_decay_theta
inference = args.inference
cov_rank = args.cov_rank

## active learning params
threshold = args.threshold
budget = args.budget
batch_size = args.batch_size
acq = args.acq
coreset = args.coreset
num_projections = args.num_projections
num_features = args.num_features
gamma = args.gamma
num_samples_IS = args.num_samples_IS
num_sample_Inner_BatchBALD = args.num_sample_Inner_BatchBALD
num_inference_samples = args.num_inference_samples
temperature = args.temperature





if acq != "Sparse Subset Approximation" and acq != "BALD-FW" and acq != "Entropy-FW-SSA" and acq != "alpha_divergence" and acq != "Thompson_BALD" and acq!= "Thompson_Likelihood_BALD" and acq!= "FULL_Thompson_BALD" and acq!= "Boltzmann_BALD":
    num_projections = None
    gamma = None
    coreset = ""
    


if acq not in ["Sparse Subset Approximation", "BALD", "BatchBALD", "Random", "BALD-FW", "Entropy-FW-SSA", "BatchBALD_variant", "BALD-FW-modified", "alpha_divergence", "Thompson_BALD", "Thompson_Likelihood_BALD", "FULL_Thompson_BALD", "Boltzmann_BALD"]:
    raise ValueError('Invalid inference method: {}'.format(acq))

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"


##
utils.set_gpu_mode(True)
pretrained_model = False

np.random.seed(seed)

torch.manual_seed(seed)
num_test_points = 10000
if dataset == 'fashion_mnist':
    from src.Bayesian_coresets.acs.al_data_set import mnist_train_transform as train_transform, \
        mnist_test_transform as test_transform
elif dataset == 'digits_mnist':
    from src.Bayesian_coresets.acs.al_data_set import mnist_train_transform as train_transform, \
        mnist_test_transform as test_transform
else:
    from src.Bayesian_coresets.acs.al_data_set import torchvision_train_transform as train_transform, \
        torchvision_test_transform as test_transform

    if dataset == 'svhn':
        num_test_points = 26032

model = resnet18(pretrained=pretrained_model, pretrained_model_file=model_file, resnet_size=84)
model = utils.to_gpu(model)
dataset = utils.get_torchvision_dataset(
    name=dataset,
    data_dir=data_dir,
    model=model,
    encode=False,  # if pretrained model True, encdoe True
    seed=seed,
    n_split=(-1, 10000, num_test_points),
    num_repetitions=num_repetitions
)

init_num_labeled = len(dataset[1]['train']) if coreset == 'Best' else init_num_labeled
data = ALD(dataset, init_num_labeled=init_num_labeled, normalize=False)



title_str = '{} {} {} (Initial # of Labeled={}, Batch size={}, J={}, g={}, budget={}, threshold={}, # repeatitions={}, num_samples_IS={}, num_inference_samples={}, temperature={})'.format(acq, coreset, args.dataset, init_num_labeled, batch_size, num_projections, gamma, budget, threshold, num_repetitions,num_samples_IS, num_inference_samples,temperature)

optim_params = {'num_epochs': training_epochs, 'batch_size': 128, 'initial_lr': initial_lr,
                'weight_decay': weight_decay, 'weight_decay_theta': weight_decay_theta,
                'train_transform': train_transform, 'val_transform': test_transform}
kwargs = {'metric': 'Acc', 'feature_extractor': model, 'num_features': num_features}
cs_kwargs = {'gamma': gamma}
print("Optimization batch size {}".format(128))

if inference in ['MF', 'Full']:
    from src.model import NeuralClassification

    kwargs['full_cov'] = inference == 'Full'
    kwargs['cov_rank'] = cov_rank
elif inference == 'MCDropout':
    from src.Bayesian_coresets.acs.model import NeuralClassificationMCDropout as NeuralClassification
else:
    raise ValueError('Invalid inference method: {}'.format(inference))

print('==============================================================================================')
print(title_str)
print('==============================================================================================')


if coreset == 'FW':
    coreset = ProjectedFrankWolfe


test_performances = {'LL': [], 'Acc': [], 'ppos': [], 'wt': [], 'num_samples': [], 'batch': [], 'scores': [], 'total_scores': [], 'batch_bald_scores': []}
test_nll, test_performance = np.zeros(1, ), np.zeros(1, )

start_time = time.time()
while len(data.index['train']) < init_num_labeled + budget:
    print('{}: Number of samples {}/{}'.format(
        seed, len(data.index['train']) - init_num_labeled, budget))
    nl = NeuralClassification(data, **kwargs)
    nl = utils.to_gpu(nl)
    nl.optimize(data, **optim_params)
    wall_time = time.time() - start_time
    num_samples = len(data.index['train']) - init_num_labeled
    test_nll, test_performance = nl.test(data, transform=test_transform)

    #============================#
    #====Active Learning step====#
    #============================#

    if acq == "BALD":
        logits_N_K_C = nl.get_pool_predictions_NKC(data, num_inference_samples= num_inference_samples, transform=test_transform)
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            batch, total_scores = batchbald.get_bald_batch(logits_N_K_C, batch_size)
        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch.indices)
        print("batch", dataset_indices)
        print("Scores: ", batch.scores)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]
        test_performances['batch'].append(dataset_indices)
        test_performances['total_scores'].append(total_scores)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))


    elif acq == "Sparse Subset Approximation":
        cs = coreset(nl, data, num_projections, acq_type="Sparse Subset Approximation", transform=test_transform, **cs_kwargs)
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        batch = cs.build(batch_size)
        # print(batch)
        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch)
        print("batch", dataset_indices)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]

        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))
        test_performances['batch'].append(dataset_indices)

    elif acq == "BALD-FW":
        logits_N_K_C = nl.get_pool_predictions_NKC(data, transform=test_transform)
        # batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            bald = batchbald.get_bald(logits_N_K_C.exp_())
        # cs_kwargs['batch'] = batch
        cs = coreset(nl, data, num_projections, batch_size=batch_size, acq_type="BALD-FW", threshold=threshold, bald=bald, transform=test_transform, **cs_kwargs)
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        batch = cs.build(batch_size)
        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch)
        # candiate_scores, candidate_indices = torch.topk(bald, 300)
        bald_scores, bald_socres_idx = bald[batch].sort()
        print("batch", dataset_indices)
        print("scores: ", np.flip(bald_scores.numpy()).tolist())
        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]

        test_performances['total_scores'].append(bald)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))
        test_performances['batch'].append(dataset_indices)
        test_performances['scores'].append(bald_scores.numpy())


    elif acq == "BatchBALD":
        logits_N_K_C = nl.get_pool_predictions_NKC(data, transform=test_transform)
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            batch = batchbald.get_batchbald_batch(logits_N_K_C,
                                                            batch_size,
                                                            num_samples_IS,
                                                              dtype=torch.double,
                                                              device=device
                                                            )
        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch.indices)
        print("batch", dataset_indices)
        print("Scores: ", batch.scores)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]
        test_performances['batch'].append(dataset_indices)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))

    elif acq == "Entropy-FW-SSA":
        logits_N_K_C = nl.get_pool_predictions_NKC(data, transform=test_transform)
        with torch.no_grad():
            entropies_N_K = batchbald.comp_cond_entropy(logits_N_K_C)



        cs = coreset(nl, data, num_projections, batch_size = batch_size, acq_type="Entropy-FW-SSA", bald=entropies_N_K, transform=test_transform, **cs_kwargs)
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        batch = cs.build(batch_size)
        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch)
        # candiate_scores, candidate_indices = torch.topk(bald, 300)
        # bald_scores, bald_socres_idx = bald[batch].sort()
        print("batch", dataset_indices)
        # print("scores: ", np.flip(bald_scores.numpy()).tolist())
        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]

        # test_performances['total_scores'].append(bald)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))
        test_performances['batch'].append(dataset_indices)
        # test_performances['scores'].append(bald_scores.numpy())

    elif acq == "BatchBALD_variant":
        logits_N_K_C = nl.get_pool_predictions_NKC(data, transform=test_transform)
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            batch = batchbald.get_batchbald_two_batch(logits_N_K_C,
                                                            batch_size,
                                                            100,
                                                        num_sample_Inner_BatchBALD)

        # print("batch:", batch)
        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch.indices)
        print("batch", dataset_indices)
        # print("Scores: ", batch.scores)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]
        test_performances['batch'].append(dataset_indices)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))




    elif acq == "BALD-FW-modified":
        logits_N_K_C = nl.get_pool_predictions_NKC(data, transform=test_transform)
        logits_N_K_C[logits_N_K_C == 0.]

        N, K, C = (logits_N_K_C).shape
        cross_prods = torch.empty((N, N), dtype=torch.float16).cuda()
        mean_probs_n_C = logits_N_K_C.mean(dim=1)

        conditional = (torch.sum(logits_N_K_C * torch.log(logits_N_K_C), dim=(1, 2)) / K)

        for i in tqdm(range(N), desc="calculate cross ...", leave=False):
            cross_prods[i, :] = -(((mean_probs_n_C * torch.log(mean_probs_n_C[i])) + (
                        mean_probs_n_C[i] * torch.log(mean_probs_n_C))) / 2).sum(1)
            cross_prods[i, i] += conditional[i]
        with torch.no_grad():
            bald = batchbald.get_bald(logits_N_K_C)

        candiate_scores, candidate_indices = torch.topk(torch.sqrt(bald) * cross_prods.sum(1).cpu(), batch_size)
        candiate_scores, candidate_indices = candiate_scores.tolist(), candidate_indices.tolist()
        bald_like_score, bald_like_idx = bald[candidate_indices].flip(0).sort()


        ######
        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(candidate_indices)
        # candiate_scores, candidate_indices = torch.topk(bald, 300)
        # bald_scores, bald_socres_idx = bald[bald_like_idx].sort()
        print("batch", dataset_indices)
        print("Scores: ", bald_like_score.flip(0).tolist())
        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]

        test_performances['total_scores'].append(bald)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))
        test_performances['batch'].append(dataset_indices)
        test_performances['scores'].append(bald_like_score.numpy())

        ######
    if acq == "alpha_divergence":
        logits_N_K_C = nl.get_pool_predictions_NKC(data, transform=test_transform)
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            batch, total_scores = batchbald.alpha_divergence(logits_N_K_C, batch_size)
        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch.indices)
        print("batch", dataset_indices)
        print("Scores: ", batch.scores)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]
        test_performances['batch'].append(dataset_indices)
        test_performances['total_scores'].append(total_scores)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))






    elif acq == "Thompson_BALD":
        hc = lambda l: torch.distributions.Categorical(logits=l)
        feat_x = []
        feature_extractor_output = []
        with torch.no_grad():
            dataloader = DataLoader(Dataset(data, 'unlabeled', transform=test_transform),
                                    batch_size=256, shuffle=False)

            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(nl.encode(x))
                feature_extractor_output.append(nl.feature_extractor(x))
            feat_x = torch.cat(feat_x)  # shape: pool_size x num_features

            KNC = hc(nl.linear(feat_x, num_samples=300))
            logits_N_K_C = KNC.probs.permute((1, 0, 2))
        logits_N_K_C = logits_N_K_C.cpu()
        logits_N_K_C[logits_N_K_C == 0] = 1e-12
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            batch, total_scores, batch_bald_scores = batchbald.thommpson_bald(logits_N_K_C, batch_size)

        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch.indices)
        print("batch", dataset_indices)
        print("Scores: ", batch.scores)
        print("Bald scores: ", batch_bald_scores)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]
        test_performances['batch'].append(dataset_indices)
        test_performances['total_scores'].append(total_scores)
        test_performances['batch_bald_scores'].append(batch_bald_scores)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))



    elif acq == "Boltzmann_BALD":
        hc = lambda l: torch.distributions.Categorical(logits=l)
        feat_x = []
        feature_extractor_output = []
        with torch.no_grad():
            dataloader = DataLoader(Dataset(data, 'unlabeled', transform=test_transform),
                                    batch_size=256, shuffle=False)

            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(nl.encode(x))
                feature_extractor_output.append(nl.feature_extractor(x))
            feat_x = torch.cat(feat_x)  # shape: pool_size x num_features

            KNC = hc(nl.linear(feat_x, num_samples=300))
            logits_N_K_C = KNC.probs.permute((1, 0, 2))
        logits_N_K_C = logits_N_K_C.cpu()
        logits_N_K_C[logits_N_K_C == 0] = 1e-12
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            batch = batchbald.Boltzmann_BALD(logits_N_K_C, batch_size, temperature)

        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch.indices)
        print("batch", dataset_indices)
        print("Scores: ", batch.scores)
        # print("Bald scores: ", batch_bald_scores)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]
        test_performances['batch'].append(dataset_indices)
        # test_performances['total_scores'].append(total_scores)
        # test_performances['batch_bald_scores'].append(batch_bald_scores)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))



    elif acq == "FULL_Thompson_BALD":
        hc = lambda l: torch.distributions.Categorical(logits=l)
        feat_x = []
        feature_extractor_output = []
        with torch.no_grad():
            dataloader = DataLoader(Dataset(data, 'unlabeled', transform=test_transform),
                                    batch_size=256, shuffle=False)

            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(nl.encode(x))
                feature_extractor_output.append(nl.feature_extractor(x))
            feat_x = torch.cat(feat_x)  # shape: pool_size x num_features

            KNC = hc(nl.linear(feat_x, num_samples=300))
            logits_N_K_C = KNC.probs.permute((1, 0, 2))
        logits_N_K_C = logits_N_K_C.cpu()
        logits_N_K_C[logits_N_K_C == 0] = 1e-12
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            batch, total_scores, batch_bald_scores = batchbald.FULL_thommpson_bald(logits_N_K_C, batch_size)

        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch.indices)
        print("batch", dataset_indices)
        print("Scores: ", batch.scores)
        print("Bald scores: ", batch_bald_scores)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]
        test_performances['batch'].append(dataset_indices)
        test_performances['total_scores'].append(total_scores)
        test_performances['batch_bald_scores'].append(batch_bald_scores)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))










    elif acq == "Thompson_Likelihood_BALD":
        hc = lambda l: torch.distributions.Categorical(logits=l)
        feat_x = []
        feature_extractor_output = []
        with torch.no_grad():
            dataloader = DataLoader(Dataset(data, 'unlabeled', transform=test_transform),
                                    batch_size=256, shuffle=False)

            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(nl.encode(x))
                feature_extractor_output.append(nl.feature_extractor(x))
            feat_x = torch.cat(feat_x)  # shape: pool_size x num_features

            KNC = hc(nl.linear(feat_x, num_samples=1000))
            logits_N_K_C = KNC.probs.permute((1, 0, 2))
        logits_N_K_C = logits_N_K_C.cpu()
        logits_N_K_C[logits_N_K_C == 0] = 1e-12
        batch_size = min(batch_size, init_num_labeled + budget - len(data.index['train']))
        with torch.no_grad():
            batch, total_scores = batchbald.thommpson_likelihood_bald(logits_N_K_C, batch_size)

        _, y_q, dataset_indices = data.move_from_unlabeled_to_train(batch.indices)
        print("batch", dataset_indices)
        print("Scores: ", batch.scores)

        class_counts = np.bincount(y_q.flatten())
        idx = np.nonzero(class_counts)[0]
        test_performances['batch'].append(dataset_indices)
        test_performances['total_scores'].append(total_scores)
        print('Class counts: {}'.format(list(zip(idx, class_counts[idx]))))




    elif acq == "Random":
        dataloader = DataLoader(Dataset(data, 'prediction', x_star=data.X, transform=test_transform),
                                batch_size=1024, shuffle=False)

        feat_data = deepcopy(data)
        feat_x = []
        with torch.no_grad():
            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(nl.encode(x))

            feat_data.X = torch.cat(feat_x)
            if args.coreset in ['KMedoids', 'KCenter']:
                feat_data.X = feat_data.X.cpu().numpy()

            def post(X, y, **kwargs):
                mean, cov = nl.linear._compute_posterior()
                return mean.cpu().detach().numpy(), cov.cpu().detach().numpy()

            cs = Random(acq, feat_data, post, save_dir=save_dir, **cs_kwargs)

        batch_size = min(args.batch_size, args.init_num_labeled + args.budget - len(data.index['train']))
        batch = cs.build(batch_size)
        data.move_from_unlabeled_to_train(batch)




    print()







    test_performances['num_samples'].append(num_samples)
    test_performances['wt'].append(wall_time)
    test_performances['ppos'].append(1 - np.mean(data.y[data.index['train']]))
    test_performances['LL'].append(-test_nll.mean())
    test_performances['Acc'].append(test_performance.mean())
    if acq not in ["Sparse Subset Approximation", "Random", "BALD-FW", "Entropy-FW-SSA","BALD-FW-modified"]:
        test_performances['scores'].append(batch.scores)

nl = NeuralClassification(data, **kwargs)
nl = utils.to_gpu(nl)
nl.optimize(data, **optim_params)

wall_time = time.time() - start_time
train_idx = np.array(data.index['train'])
print(train_idx[init_num_labeled:])
num_samples = len(train_idx) - init_num_labeled
test_nll, test_performance = nl.test(data, transform=test_transform)

test_performances['num_samples'].append(num_samples)
test_performances['wt'].append(wall_time)
test_performances['ppos'].append(1 - np.mean(data.y[train_idx]))
test_performances['LL'].append(-test_nll.mean())
test_performances['Acc'].append(test_performance.mean())
test_performances['num_evals'] = np.arange(len(test_performances['Acc']) + 1)
test_performances['init_num_labeled'] = init_num_labeled
test_performances['train_idx'] = train_idx
test_performances['wt'][0] = 0.

if coreset == 'Best':
    test_performances['num_samples'].append(budget)
    test_performances['wt'].append(wall_time)
    test_performances['ppos'].append(1 - np.mean(data.y[train_idx]))
    test_performances['LL'].append(-test_nll.mean())
    test_performances['Acc'].append(test_performance.mean())

with open(os.path.join(save_dir, title_str + "_" + result_file_name_index + '.pkl'), 'wb') as handle:
    pickle.dump({title_str: test_performances}, handle)


# with open(os.path.join(save_dir, result_file_name + '.pkl'), "rb") as input_file:
#     test_performances = pickle.load(input_file)

import gtimer as gt

from torch.nn import functional as F
from src.BatchBALD import consistent_mc_dropout
import torch.utils.model_zoo as model_zoo

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

import src.Bayesian_coresets.acs.utils as utils
from src.Bayesian_coresets.acs.al_data_set import Dataset


from src import batchbald


# import consistent_mc_dropout

class BayesianCNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, inplanes=64, initial_kernel_size=7, initial_pool=True, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.initial_pool = initial_pool
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=initial_kernel_size, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if initial_kernel_size != 5:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, param_dict=None, num_samples=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def get_layer_output(self, x, param_dict, layer_to_return):
        if layer_to_return == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if self.initial_pool:
                x = self.maxpool(x)
            return x
        else:
            resnet_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            layer = layer_to_return - 1
            for block in range(self.layers[layer]):
                x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'], param_dict[layer][block]['beta1'],
                                       param_dict[layer][block]['gamma2'], param_dict[layer][block]['beta2'])
            return x

    @property
    def output_size(self):
        return 512


def resnet18(pretrained=False, pretrained_model_file='./model_best.pth.tar', resnet_size=224, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        large (bool): Additional parameter for calling ResNet for 84x84 images (if false)
    """
    if resnet_size == 84:
        model = ResNet(BasicBlock, [2, 2, 2, 2], inplanes=64, initial_kernel_size=5, initial_pool=False, **kwargs)
        if pretrained:
            ckpt_dict = torch.load(pretrained_model_file)
            model.load_state_dict(ckpt_dict['state_dict'])
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    model.pretrained = pretrained
    return model


class LinearVariance(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        """
        Helper module for computing the variance given a linear layer.
        :param in_features: (int) Number of input features to layer.
        :param out_features: (int) Number of output features from layer.
        """
        super().__init__(in_features, out_features, bias)
        self.softplus = nn.Softplus()

    @property
    def w_var(self):
        """
        Computes variance from log std parameter.
        :return: (torch.tensor) Variance
        """
        return self.softplus(self.weight) ** 2

    def forward(self, x):
        """
        Computes a forward pass through the layer with the squared values of the inputs.
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Variance of predictions
        """
        return torch.nn.functional.linear(x ** 2, self.w_var, bias=self.bias)



class LocalReparamDense(nn.Module):
    def __init__(self, shape):
        """
        A wrapper module for functional dense layer that performs local reparametrization.
        :param shape: ((int, int) tuple) Number of input / output features to layer.
        """
        super().__init__()
        self.in_features, self.out_features = shape
        self.mean = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=True
        )

        self.var = LinearVariance(self.in_features, self.out_features, bias=False)

        nn.init.normal_(self.mean.weight, 0., 0.05)
        nn.init.normal_(self.var.weight, -4., 0.05)

    def forward(self, x, num_samples=1, squeeze=False):
        """
        Computes a forward pass through the layer.
        :param x: (torch.tensor) Inputs.
        :param num_samples: (int) Number of samples to take.
        :param squeeze: (bool) Squeeze unnecessary dimensions.
        :return: (torch.tensor) Reparametrized sample from the layer.
        """
        mean, var = self.mean(x), self.var(x)
        return utils.sample_normal(mean, var, num_samples, squeeze)

    def compute_kl(self):
        """
        Computes the KL divergence w.r.t. a standard Normal prior.
        :return: (torch.tensor) KL divergence value.
        """
        mean, cov = self._compute_posterior()
        scale = 2. / self.mean.weight.shape[0]
        # scale = 1.
        return utils.gaussian_kl_diag(mean, torch.diag(cov), torch.zeros_like(mean), scale * torch.ones_like(mean))

    def _compute_posterior(self):
        """
        Returns the approximate posterior over the weights.
        :return: (torch.tensor, torch.tensor) Posterior mean and covariance for layer weights.
        """
        return self.mean.weight.flatten(), torch.diag(self.var.w_var.flatten())


class ReparamFullDense(nn.Module):
    def __init__(self, shape, bias=True, rank=None):
        """
        Reparameterization module for dense covariance layer.
        :param shape: ((int, int) tuple) Number of input / output features.
        :param bias: (bool) Use a bias term in the layer.
        :param rank: (int) Rank of covariance matrix approximation.
        """
        super().__init__()
        self.in_features, self.out_features = shape
        self.mean = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias
        )

        # Initialize (possibly low-rank) covariance matrix
        covariance_shape = np.prod(shape)
        rank = covariance_shape if rank is None else rank
        self.F = torch.nn.Parameter(torch.zeros(covariance_shape, rank))
        self.log_std = torch.nn.Parameter(torch.zeros(covariance_shape))
        nn.init.normal_(self.mean.weight, 0., 0.05)
        nn.init.normal_(self.log_std, -4., 0.05)

    @property
    def variance(self):
        """
        Computes variance from log std parameter.
        :return: (torch.tensor) Variance
        """
        return torch.exp(self.log_std) ** 2

    @property
    def cov(self):
        """
        Computes covariance matrix from matrix F and variance terms.
        :return: (torch.tensor) Covariance matrix.
        """
        return self.F @ self.F.t() + torch.diag(self.variance)

    def forward(self, x, num_samples=1):
        """
        Computes a forward pass through the layer.
        :param x: (torch.tensor) Inputs.
        :param num_samples: (int) Number of samples to take.
        :return: (torch.tensor) Reparametrized sample from the layer.
        """
        mean = self.mean.weight  # need un-flattened
        post_sample = utils.sample_lr_gaussian(mean.view(1, -1), self.F, self.variance, num_samples, squeeze=True)
        post_sample = post_sample.squeeze(dim=1).view(num_samples, *mean.shape)
        return (post_sample[:, None, :, :] @ x[:, :, None].repeat(num_samples, 1, 1, 1)).squeeze(-1) + self.mean.bias

    def compute_kl(self):
        """
        Computes the KL divergence w.r.t. a standard Normal prior.
        :return: (torch.tensor) KL divergence value.
        """
        mean, cov = self._compute_posterior()
        # scale = 1.
        scale = 2. / self.mean.weight.shape[0]
        return utils.smart_gaussian_kl(mean, cov, torch.zeros_like(mean), torch.diag(scale * torch.ones_like(mean)))

    def _compute_posterior(self):
        """
        Returns the approximate posterior over the weights.
        :return: (torch.tensor, torch.tensor) Posterior mean and covariance for layer weights.
        """
        return self.mean.weight.flatten(), self.cov


class NeuralClassification(nn.Module):
    """
    Neural Linear model for multi-class classification.
    :param data: (Object) Data for model to trained / evaluated on
    :param feature_extractor: (nn.Module) Feature extractor to generate representations
    :param metric: (str) Metric to use for evaluating model
    :param num_features: (int) Dimensionality of final feature representation
    :param full_cov: (bool) Use (low-rank approximation to) full covariance matrix for last layer distribution
    :param cov_rank: (int) Optional, if using low-rank approximation, specify rank
    """

    def __init__(self, data, feature_extractor=None, metric='Acc', num_features=256, full_cov=False, cov_rank=None):
        super().__init__()
        self.num_classes = len(np.unique(data.y))
        self.feature_extractor = feature_extractor
        if self.feature_extractor.pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.num_features = num_features
        else:
            self.num_features = num_features
        self.fc1 = nn.Linear(in_features=512, out_features=self.num_features, bias=True)
        self.fc2 = nn.Linear(in_features=self.num_features, out_features=self.num_features, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        if full_cov:
            self.linear = ReparamFullDense([self.num_features, self.num_classes], rank=cov_rank)
        else:
            self.linear = LocalReparamDense([self.num_features, self.num_classes])

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.metric = metric

    def forward(self, x, num_samples=1):
        """
        Make prediction with model
        :param x: (torch.tensor) Inputs
        :param num_samples: (int) Number of samples to use in forward pass
        :return: (torch.tensor) Predictive distribution (may be tuple)
        """
        return self.linear(self.encode(x), num_samples=num_samples)

    def encode(self, x):
        """
        Use feature extractor to get features from inputs
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Feature representation of inputs
        """
        x = self.feature_extractor(x)
        x = self.fc1(x)
        x = self.relu(x)
        if self.feature_extractor.pretrained:
            x = self.fc2(x)
            x = self.relu(x)
        return x

    def optimize(self, data, num_epochs=1000, batch_size=64, initial_lr=1e-2, freq_summary=100,
                 weight_decay=1e-1, weight_decay_theta=None, train_transform=None, val_transform=None, **kwargs):
        """
        Internal functionality to train model
        :param data: (Object) Training data
        :param num_epochs: (int) Number of epochs to train for
        :param batch_size: (int) Batch-size for training
        :param initial_lr: (float) Initial learning rate
        :param weight_decay: (float) Weight-decay parameter for deterministic weights
        :param weight_decay_theta: (float) Weight-decay parameter for non-deterministic weights
        :param train_transform: (torchvision.transform) Transform procedure for training data
        :param val_transform: (torchvision.transform) Transform procedure for validation data
        :param kwargs: (dict) Optional additional arguments for optimization
        :return: None
        """
        weight_decay_theta = weight_decay if weight_decay_theta is None else weight_decay_theta
        weights = [v for k, v in self.named_parameters() if (not k.startswith('linear')) and k.endswith('weight')]
        weights_theta = [v for k, v in self.named_parameters() if k.startswith('linear') and k.endswith('weight')]
        other = [v for k, v in self.named_parameters() if not k.endswith('weight')]
        optimizer = torch.optim.Adam([
            {'params': weights, 'weight_decay': weight_decay},
            {'params': weights_theta, 'weight_decay': weight_decay_theta},
            {'params': other},
        ], lr=initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)

        dataloader = DataLoader(
            dataset=Dataset(data, 'train', transform=train_transform),  # Dataset X has shape (batch, 1, 28, 28)
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        for epoch in range(num_epochs):
            scheduler.step()
            losses, kls, performances = [], [], []
            for (x, y) in dataloader:
                optimizer.zero_grad()
                x, y = utils.to_gpu(x, y.type(torch.LongTensor).squeeze())
                y_pred = self.forward(x)  # 1 x batch_size x num_classes
                step_loss, kl = self._compute_loss(y, y_pred, len(x) / len(data.index['train']))
                step_loss.backward()
                optimizer.step()

                performance = self._evaluate_performance(y, y_pred)
                losses.append(step_loss.cpu().item())
                kls.append(kl.cpu().item())
                performances.append(performance.cpu().item())

            if epoch % freq_summary == 0 or epoch == num_epochs - 1:
                val_bsz = 1024
                val_losses, val_performances = self._evaluate(data, val_bsz, 'val', transform=val_transform, **kwargs)
                print('#{} loss: {:.4f} (val: {:.4f}), kl: {:.4f}, {}: {:.4f} (val: {:.4f})'.format(
                    epoch, np.mean(losses), np.mean(val_losses), np.mean(kls),
                    self.metric, np.mean(performances), np.mean(val_performances)))

    def get_pool_predictions_NKC(self, data, num_inference_samples=100, transform=None):
        hc = lambda l: torch.distributions.Categorical(logits=l)
        feat_x = []
        with torch.no_grad():
            mean, cov = self.linear._compute_posterior()  # mean shape: num_features x num_classes
            # cov shape: (num_features x num_classes) x (num_features x num_classes)
            jitter = utils.to_gpu(torch.eye(len(cov)) * 1e-6)


            dataloader = DataLoader(Dataset(data, 'unlabeled', transform=transform),
                                    batch_size=256, shuffle=False)

            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(self.encode(x))

            feat_x = torch.cat(feat_x)  # shape: pool_size x num_features

            KNC = hc(self.linear(feat_x, num_samples=num_inference_samples))
            # KNC = self.linear(feat_x, num_samples=100)
            # NKC = KNC.permute((1, 0, 2))

            return KNC.probs.permute((1,0,2))

    def get_projections(self, data, J, projection='two', gamma=0, transform=None, **kwargs):
        """
        Get projections for ACS approximate procedure
        :param data: (Object) Data object to get projections for
        :param J: (int) Number of projections to use
        :param projection: (str) Type of projection to use (currently only 'two' supported)
        :return: (torch.tensor) Projections
        """
        ent = lambda py: torch.distributions.Categorical(probs=py).entropy()
        projections = []
        feat_x = []
        with torch.no_grad():
            mean, cov = self.linear._compute_posterior()  # mean shape: num_features x num_classes
            # cov shape: (num_features x num_classes) x (num_features x num_classes)
            jitter = utils.to_gpu(torch.eye(len(cov)) * 1e-6)
            theta_samples = MVN(mean, cov + jitter).sample(torch.Size([J])).view(J, -1,
                                                                                 self.linear.out_features)  # shape: J x num_features x num_classes

            dataloader = DataLoader(Dataset(data, 'unlabeled', transform=transform),
                                    batch_size=256, shuffle=False)

            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(self.encode(x))

            feat_x = torch.cat(feat_x)  # shape: pool_size x num_features
            py = self._compute_predictive_posterior(self.linear(feat_x, num_samples=100),
                                                    logits=False)  # shape: pool size x num_classes
            # probability of each class

            # self.linear(feat_x, num_samples=100) shape: num_samples x pool_size x num_classes
            #                               feat_x shape: pool_size x num_features
            ent_x = ent(py)  # shape: pool_size
            if projection == 'two':
                for theta_sample in theta_samples:
                    projections.append(self._compute_expected_ll(feat_x, theta_sample, py) + gamma * ent_x[:,
                                                                                                     None])  # shape: pool size x 1
                    # L_m(theta) but without H[y_m|x_m,D_0]
            else:
                raise NotImplementedError
        # torch.cat(projections, dim=1) shape: pool size x num_projections

        return utils.to_gpu(torch.sqrt(1 / torch.FloatTensor([J]))) * torch.cat(projections, dim=1), ent_x

    def test(self, data, **kwargs):
        """
        Test model
        :param data: (Object) Data to use for testing
        :param kwargs: (dict) Optional additional arguments for testing
        :return: (np.array) Performance metrics evaluated for testing
        """
        print("Testing...")

        # test_bsz = len(data.index['test'])
        test_bsz = 1024
        losses, performances = self._evaluate(data, test_bsz, 'test', **kwargs)
        print("predictive ll: {:.4f}, N: {}, {}: {:.4f}".format(
            -np.mean(losses), len(data.index['train']), self.metric, np.mean(performances)))
        return np.hstack(losses), np.hstack(performances)

    def _compute_log_likelihood(self, y, y_pred):
        """
        Compute log-likelihood of predictions (this is loss; ELBO can be achieved by summing this loss and minus KL)
        :param y: (torch.tensor) Observations :: shape: batch_size
        :param y_pred: (torch.tensor) Predictions
        :return: (torch.tensor) Log-likelihood of predictions
        """
        log_pred_samples = y_pred  # shape: 1 x batch_size x num_classes
        ll_samples = torch.stack([-self.cross_entropy(logit, y) for logit in log_pred_samples])  # shape: 1 x batch_size
        # logit shape: batch_size x num_classes
        # y shape: batch_size (true label y)
        # torch.mean(ll_samples, dim=0) shape: batch_size
        return torch.sum(torch.mean(ll_samples, dim=0), dim=0)

    def _compute_predictive_posterior(self, y_pred, logits=True):
        """
        Return posterior predictive evaluated at x
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Probit regression posterior predictive
        """
        log_pred_samples = y_pred  # num_samples x batch_size x num_classes
        L = utils.to_gpu(torch.FloatTensor([log_pred_samples.shape[0]]))  # num_samples
        preds = torch.logsumexp(log_pred_samples, dim=0) - torch.log(L)  # batch_size x num_classes
        if not logits:
            preds = torch.softmax(preds, dim=-1)
        return preds

    def _compute_loss(self, y, y_pred, kl_scale=None):
        """
        Compute loss function for variational training
        :param y: (torch.tensor) Observations
        :param y_pred: (torch.tensor) Model predictions
        :param kl_scale: (float) Scaling factor for KL-term
        :return: (torch.scalar) Loss evaluation
        """
        # The objective is 1/n * (\sum_i log_like_i - KL)
        log_likelihood = self._compute_log_likelihood(y, y_pred)
        kl = self.linear.compute_kl() * kl_scale
        elbo = log_likelihood - kl
        return -elbo, kl

    def _compute_expected_ll(self, x, theta, py):
        """
        Compute expected log-likelihood for data
        :param x: (torch.tensor) Inputs to compute likelihood for
        :param theta: (torch.tensor) Theta parameter to use in likelihood computations
        :return: (torch.tensor) Expected log-likelihood of inputs
        """
        logits = x @ theta  # shape: pool_size x num_classes
        # x shape: pool_size x num_features
        # theta shape: num_features x num_classes
        ys = torch.ones_like(logits).type(torch.LongTensor) * torch.arange(self.linear.out_features)[None, :]
        ys = utils.to_gpu(ys).t()  # shape: num_classes x pool_size

        loglik = torch.stack([-self.cross_entropy(logits, y) for y in ys]).t()  # shape: pool_size x num_classes
        # logits shape: pool_size x num_classes
        # y shape: pool_size

        # py shape: pool size x num_classes (probability of each class)
        return torch.sum(py * loglik, dim=-1, keepdim=True)  # shape: pool_size x 1

    def _evaluate_performance(self, y, y_pred):
        """
        Evaluate performance metric for model
        """
        log_pred_samples = y_pred
        y2 = self._compute_predictive_posterior(log_pred_samples)
        return torch.mean((y == torch.argmax(y2, dim=-1)).float())

    def _evaluate(self, data, batch_size, data_type='test', transform=None):
        """
        Evaluate model with data
        :param data: (Object) Data to use for evaluation
        :param batch_size: (int) Batch-size for evaluation procedure (memory issues)
        :param data_type: (str) Data split to use for evaluation
        :param transform: (torchvision.transform) Tranform procedure applied to data during training / validation
        :return: (np.arrays) Performance metrics for model
        """
        assert data_type in ['val', 'test']
        losses, performances = [], []

        if data_type == 'val' and len(data.index['val']) == 0:
            return losses, performances

        gt.pause()
        with torch.no_grad():
            dataloader = DataLoader(
                dataset=Dataset(data, data_type, transform=transform),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4
            )
            for (x, y) in dataloader:
                x, y = utils.to_gpu(x, y.type(torch.LongTensor).squeeze())
                y_pred_samples = self.forward(x, num_samples=100)
                y_pred = self._compute_predictive_posterior(y_pred_samples)[None, :, :]
                loss = self._compute_log_likelihood(y, y_pred)  # use predictive at test time
                avg_loss = loss / len(x)
                performance = self._evaluate_performance(y, y_pred_samples)
                losses.append(avg_loss.cpu().item())
                performances.append(performance.cpu().item())

        gt.resume()
        return losses, performances





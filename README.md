# ThompsonBALD: Bayesian Batch Active Learning for Deep Learning via Thompson Sampling

## Abstract
Deep learning has been successfully applied to many pattern recognition tasks, but typically requires large labelled datasets and presents challenges in domains where acquiring labels is expensive. 
Probabilistic active learning methods aim to help by framing the labeling process as a decision problem, greedily selecting the most informative next data point to label. 
However, difficulties arise when considering a batch active learning setting: naive greedy approaches to batch construction can result in highly correlated queries. 
In this work, we introduce ThompsonBALD, a simple and surprisingly effective Bayesian batch active learning method, based on Thompson sampling of the mutual information between a data point and the model parameters.
We demonstrate ThompsonBALD achieves performance comparable to other recent methods with significantly reduced computational time,
and also compares favorably to other approaches which achieve batch diversity through injecting noise.

## Links
<a href="https://github.com/jaeikjeon9919/ThompsonBALD-Bayesian-Batch-Active-Learning/raw/master/MSc_Dissertation.pdf" download="ThompsonBALD_bayesian_batch_active_learning_for_deep_learning_via_thompson_sampling.pdf">MSc Dissertation</a>,
<a href="https://github.com/jaeikjeon9919/ThompsonBALD-Bayesian-Batch-Active-Learning/raw/master/ThompsonBALD_bayesian_batch_active_learning_for_deep_learning_via_thompson_sampling.pdf" download="ThompsonBALD_bayesian_batch_active_learning_for_deep_learning_via_thompson_sampling_poster.pdf">Poster for NeurIPS Europe meetup on
Bayesian Deep Learning</a>

# ThompsonBALD: Bayesian Batch Active Learning for Deep Learning via Thompson Sampling

MNIST test accuracy as a function of number of acquired images from the pool set.
-   Initial training set of 20 data points
-   Bald averaged over 5 repetitions
-   Batch-bald averaged over 3 repetitions.
-   The shaded area represents 1 s.d.
-   No early stopping
-   acquire 5 points each time from the pool set
![GitHub Logo](figs/BatchBald_vs_Bald_5points_aquired.png)



# 18.06.2020 Update

## t-SNE implementation for MNIST (BALD, BatchBALD, Sparse Subset Approximation)
Corresponding jupyter notebook file: [t-SNE_plot](t-SNE_plot.ipynb)
### When the training starts from 20 data. (it has little prediction power i.e. 60-70% accuracy)

#### BASELINE: acquire 1 data at a time with BALD, total of 70 data.

![GitHub Logo](figs/BASELINE_with_BALD_from_20_data.png)


#### BALD: acquire 70 data at a time, total of 70 data

![GitHub Logo](figs/BALD_from_20_data.png)

#### BatchBALD: acquire 50 data at a time, total of 50 data

![GitHub Logo](figs/BatchBALD_from_20_data_acquire_50.png)

#### SPARSE SUBSET APPROXIMATION: acquire 70 data at a time, total of 70 data

![GitHub Logo](figs/SSA_from_20_data_acquire_70.png)



### When the training starts from 300 data. (it has strong prediction power i.e. 92% accuracy)

#### BASELINE: acquire 1 data at a time with BALD, total of 70 data.

![GitHub Logo](figs/BASELINE_with_BALD_from_300_data.png)

#### BALD: acquire 70 data at a time, total of 70 data

![GitHub Logo](figs/BALD_from_20_data_acquire_70.png)

Batching 50 data with BatchBALD takes more than 7 hours. I had no time to implement this. Instead, compare with batching 15 data.

#### BALD: acquire 15 data at a time, total of 15 data

![GitHub Logo](figs/BALD_from_20_data_acquire_15.png)


#### BatchBALD: acquire 15 data at a time, total of 15 data

![GitHub Logo](figs/BatchBALD_from_20_data_acquire_15.png)

cannot notice significant difference...


### When the training starts from 20 data, but now with repeated MNIST (replication of 60000 MNIST digits 3 times with adding some noise)

#### BASELINE: acquire 1 data at a time with BALD, total of 70 data.

![GitHub Logo](figs/BASELINE_with_BALD_from_20_data_REPATED_MNIST.png)

#### BALD: acquire 70 data at a time, total of 70 data

![GitHub Logo](figs/BALD_from_20_data_acquire_70_REPEATED_MNIST.png)

Note that the number of acquired data looks far less than 70. This means BALD selects same digits with overlapping.

#### BatchBALD: acquire 70 data at a time, total of 70 data

Again, had no time to implment this, but expect to have way better result than BALD, based on figure 4 of BatchBALD paper.

### Now, 5 data is acquired from pool set with maximum training sample 70. Correlations of each batches are compared by model's predcition power

#### BALD: 60%, 90%, 95% accuracy

![GitHub Logo](figs/BALD_ACC_60p_70_samples_batch_5.png)

![GitHub Logo](figs/BALD_ACC_90p_70_samples_batch_5.png)

![GitHub Logo](figs/BALD_ACC_95p_70_samples_batch_5.png)

#### BatchBALD: 60%, 90%, 95% accuracy

![GitHub Logo](figs/BatchBALD_ACC_60p_70_samples_batch_5.png)

![GitHub Logo](figs/BatchBALD_ACC_90p_70_samples_batch_5.png)

![GitHub Logo](figs/BALD_ACC_95p_70_samples_batch_5.png)

#### Note that there is no correlation between all batches. This explains why there is no significant difference in accuracy between BatchBALD and BALD in the very first figure.

# 25.06.2020 Update
## BALD, Neural-Linear, batch size 100
![GitHub Logo](figs/Neural_Linear_100batch_bald_1.png)

## BatchBALD, Neural-Linear, batch size 100
![GitHub Logo](figs/Neural_Linear_100batch_batchbald_1.png)



## Coreset-Sparse Subset Approximation, Neural-Linear, batch size 100
![GitHub Logo](figs/Neural_Linear_100batch_SSA_1.png)



## 
![GitHub Logo](figs/All_plot.png)



## Neural Linear
![GitHub Logo](figs/neural-linear.png)



## MLP
![GitHub Logo](figs/MLP.png)




# 29.06.2020 Update




# 1. Introduction

Pole distribution is a feature to characterise the linear time-invariant (LTI) system.
According to the pole distribution of the system,
the different dynamism is observed in the time scaled trend.

Linearly recurrent network (LRN) system is a branch
of the sequential model in the machine learning.
Networks can be trained base on datasets 
by using machine learning algorithms.

We study empirically how trained LRN systems
estimate the pole distribution of targeted LTI systems
and how to improve the estimation by tuning training hyperparameters.

# 2. Specifications


## loss function

Agents are trained by the following criteria:

<img src = "./img/texclip20200719102549.png" width = "83%">

<img src = "./img/texclip20200719102549.png">

,where `y(t)` denotes the observation at the time `t` 
and `yhat(t|s)` denotes the prediction at the time `t (t>s)`
given the observation at the time `s` and the input series between `s` and `t`.
The time span between `s` and `s + Nhrz` is called the prediction horizon 
at the query of prediction at the time `s`


# 3. Case studies

editing

## 3-1. Case study #1:

In this case study,
LRN systems were trained by using the hyperparameters in the table 3.1.1.
The figure 3.1.1 shows the learning curves
of the discrepancy between the true pole distribution
and the estimated one defined as follows:

<img src = "./img/texclip20200718160550.png" width = "83%">

It's confirmed that the estimated errors have converted 
at the end of training iterations.

The figure 3.1.2 shows some examples of the pair of
the targeted pole distribution and the trained one,
which are selected randomly among the trained networks.
It can be seen that all the poles are missed
and that estimated poles are not always but sometimes located beyond the unit circle,
which means that the estimated LRN system might be unstable.

<img src = "./img/pole_distribution_discrepancy_case_study_001a.png" width = "50%">
Figure 3.1.1 Learning curves of the discrepancy between the true pole distribution and the estimated one

<img src = "./img/pole_distribution_examples_case_study_001a.png" width = "50%">
Figure 3.1.2 Targeted pole distributions and trained ones

## 3-2. Case study #2:

Generally speaking,
time constant of environment
is longer than sampling interval,
which implies that
, if an agent is allowed to see the prediction at the next step,
it might be hard for the agent to learn the entire response of the environment.
Intuitively, the prediction horizon should be similar to or longer than
the time constant of the environment.

In this case study,
we choose the prediction horizon among the options of 1,4 and 16,
as shown in the table 3.2.1,
since the targeted environment has time constants distributed between 4 and 32.

Figure 3.2.1 shows the learning curves
of the pole distribution discrepancy, already  mentioned in the case study 1.
It's confirmed that
- the discrepancy have almost converged at the end of the training iterations,
- and the mismatches of the pole distribution reduce depending on the length of the prediction horizon.

Figure 3.2.2 show some examples of pole distributions of randomly selected trained agents.
Some poles of trained systems match with the poles of targeted system
or are located in the center of groups of the targeted poles as if they represented the groups.
The rest of poles of the trained systems are apart from the unit circle,
this means that the trained system does not have the capacity
to estimate long-term behaviour of the targeted system.

<img src = "./img/pole_distribution_discrepancy_case_study_002a.png" width = "50%">
Figure 3.2.1 Learning curves of the discrepancy between the true pole distribution and the estimated one

<img src = "./img/pole_distribution_examples_case_study_002a.png" width = "50%">
Figure 3.2.2 Targeted pole distributions and trained ones

## 3-3. Case study #3:

As mentioned in the discussion of the case study #2,
trained system can loose the capacity of the long-term prediction capacity.
This might be because the random initialization of the system matrix of agents
randomly allocates its poles on the complex plain which does not always have the position near the unit circle.
Therefore, this case study assign the poles inside the unit circle but close to the edge
in the initialization by exploiting the canonical decomposition of the linear system.

Agents are parameterized as follows:
x_{i}(t+1) = lambda_{i} * x_{i}(t) + sum(b_{ij} * w_{j}(t),j), i = 0:n,
y(t) = C * x(t) + c.c..
And, lambda_{i} and b_{ij} are initialized like this:
|lambda_{i}| = d, b_{ij} ~ N(0, sigma^2), sigma^2 = 1-d^2,
where d denote the initial damping constant, a given parameter.

The figure 3.3.1 show the learning curves of the performance defined in the case study 1.
Thanks for the random allocation of the initial poles on the edge of the unit circle,
the performances converge faster than the agents with the system matrix initialized randomly. 
Furthermore, the variance of the error is shrank, though,
the average performance does not has big difference between two types of agents.
The latter is understandable since it's normal that neither the diffrent expressions of the system
nor the different initialization of the poles
can improve the estimation capacity of the pole distribution.

The figure 3.3 shows some examples of estimated pole distribution.
In so much as these examples,
there is no unrelevant poles of the trained system.
No pole is located apart from the unit circle
or from the targeted pole distribution.

<img src = "./img/pole_distribution_discrepancy_case_study_003a.png" width = "50%">
Figure 3.3.1 Learning curves of the discrepancy between the true pole distribution and the estimated one

<img src = "./img/pole_distribution_examples_case_study_003a.png" width = "50%">
Figure 3.3.2 Targeted pole distributions and trained ones

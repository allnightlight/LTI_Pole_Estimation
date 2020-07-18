

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

editing

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
It can be seen that all the poles are missed.

<img src = "./img/pole_distribution_discrepancy_case_study_001a.png" width = "50%">
Figure 3.1.1 Learning curves of the discrepancy between the true pole distribution and the estimated one

<img src = "./img/pole_distribution_examples_case_study_001a.png" width = "50%">
Figure 3.1.2 Targeted pole distributions and trained ones

## 3-2. Case study #2:

editing
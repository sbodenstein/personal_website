---
title: "Normalizing Flows"
date: 2019-07-12
math: true
diagram: true
---


## Generative Modelling
> *"What I cannot create, I do not understand."* ~ Richard Feynman

Generative models are 



- 

{{< figure library="true" src="2020_flows/generative.png" numbered="true" title="d" lightbox="true" >}}

- Reinforcement Learning. A 
    - Planning: https://arxiv.org/abs/1807.09341


Some examples of popular generative modelling methods and what they learn:

- [Energy-based models](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf). This class of model learns the unnormalized joint distribution $\tilde{p}(\mathbf{x})$, where $p(\mathbf{x})=\tilde{p}(\mathbf{x})/\mathcal{Z}$. The normalizing constant $\mathcal{Z}$ is often intractable to compute. The advantage of this approach is that modelling an unormalized distribution $\tilde{p}(\mathbf{x})$ with a neural net is far easier than needing to add the normalization constraint. For tasks such as ranking the plausibility of different images, the normalization is not needed. There has been some promising recent work showing how to [scale this class of method](https://openai.com/blog/energy-based-models/), and using them to [improve classification tasks](https://arxiv.org/abs/1912.03263).

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (GANs). These models only allow sampling from $p(\mathbf{x})$, but explicit evaluation. If all one cares about is producing good samples, then this approach frees the net from having to learn unnecessary details. However, this also means that it is very hard to measure whether GANs overfit, as you cannot compute $p(\mathbf{x})$ on a test set.

- [Variational Autoencoders](https://arxiv.org/abs/1906.02691) (VAEs). This class of model computes a variational lower bound to $p(\mathbf{x})$. Samples can easily be drawn from $p(\mathbf{x})$, but are generally less good than those from GANs.

- [Normalizing Flows](https://arxiv.org/abs/1908.09257). These models can compute $p(\mathbf{x})$ explicitly and without approximations, and easy to draw good samples from (see **Figure 2**). The price to pay for this model being such complete approximator for $p(\mathbf{x})$ is a few restrictions on the types of neural net architectures we can use.


{{< figure library="true" src="2020_flows/glow_faces.jpg" numbered="true" title="Random samples drawn from a normalizing flow model trained on facial images. (Image source: [Kingma and Dhariwal, 2018](https://arxiv.org/abs/1807.03039))" lightbox="true" >}}


## Normalizing Flows

Building a Normalizing Flow generative model of [MNIST digits](http://yann.lecun.com/exdb/mnist/) from scratch be the motivating example for the rest of this post. Expect a strong bias towards simplicity: there will be no ConvNets, or multi-scale architectures, or other tricks needed to get the [very prettiest generated images](https://arxiv.org/abs/1807.03039).

We have dataset of images represented as a pixel-vector $\mathbf{x}$, sampled from a distribution $\mathbf{x} \sim p_\text{data}(\mathbf{x})$. This distribution is rather complicated, with highly non-trivial correlations between each of the pixels.

The idea of Normalizing Flows is to start with a *simple* distribution, one which can easily be sampled from and its PDF evaluated. This is typically a spherical multivariate Guassian $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. We then use a neural net $f$ to map samples from the simple distribution to the complicated one, $\mathbf{x} = f(\mathbf{z}) \sim p_\text{model}(\mathbf{x})$. 

We also want to be able to compute $p_\text{model}(\mathbf{x})$. The way to do this is to use the [change of variable formula for random variables](https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function). In 1-dimension, 

How to train the net $f$ so that $p_\text{model}(\mathbf{x})$ is as close as possible to $p_\text{data}(\mathbf{x})$? The idea is to maximize the expected value of $p_\text{model}(\mathbf{x})$ when evaluated on samples $\mathbf{x}$ drawn from the data distribution $p_\text{data}(\mathbf{x})$. As $p_\text{model}(\mathbf{x})$ can be very small in high-dimensions[^3], numerical stability dictates using the logarithm of $p_\text{model}(\mathbf{x})$ instead. And in machine learning, we usually minimize rather than maximize, so need a minus sign. So our loss to minimize will be:
$$
\mathcal{L} = -\frac{1}{N} \sum_{n=1}^N \log p\_\text{model} (\mathbf{x}), \\ \\ \text{where } \mathbf{x}\sim p_\text{data}
$$

But how 



In addition to drawing samples, we want to be able to compute the joint distribution with our neural net model, $p_\text{model}(\mathbf{x})$. 

How to train our net$f$? We want 

But how  $\mathbf{x}=f_\mathbf{\theta}(\mathbf{z})$ 

Once we have 

- Sampling: draw a sample of $\mathcal{z}$ from the multivariate  $\mathbf{z}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Then produce a sample of $\mathbf{x}$ using $\mathbf{x}=f_\mathbf{\theta}(\mathbf{z})$ 
- Computing $p_\mathbf{x}(\mathbf{x})$: use the inverse 
- Training: compute 

Then we want to train a neural 


 let $\mathbf{x}$ be a sample from the data distribution   $\mathbf{x} = f_\mathbf{\theta}(\mathbf{z})$, and $\mathbf{x}$  \ f_\mathbf{\theta}(\mathbf{z})$

We have some data which is sampled from the distribution $p(\mathbf{x})$, where 


let $\mathbf{x}$ by a $D$-dimensional vector. 

- A functionf:RD→RDis called a diffeomor-phism, if it is bijective, differentiable, and its inverse isdifferentiable as well.



Reviews:
- https://arxiv.org/pdf/1908.09257.pdf

### How restrictive is invertibility?

Normalizing Flows require our neural net to be invertible. This invertibility requirement is not present for other generative modelling approaches, such as GANs or VAEs. One immediate question becomes: does this invertibility requirement put significant restrictions on the type of distributions we can model using Normalizing Flows? The answer is no.

First, it can be shown that for any distribution $p_{\mathbf{x}}(\mathbf{x})$ (with some mild assumptions), there exists an invertable and differentiable function $F$ 

Finally, we also have strong empirical evidence for the expressive power of invertible neural net models. For example, i-RevNet ([Jacobsen et al., 2018](https://arxiv.org/abs/1802.07088)) is an invertible architecture achieving similar performance on ImageNet as ResNets.

### Tractable Determinants

One of the 

### Example: MNIST
https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector


#### Worked Example: BatchNorm

#### Issue: Categorical Pixel Values

Images are usually stored with 8-bit integer pixel values, $\mathbf{x} \in \\{0,\ldots,255 \\}^D$, or even boolean pixel values (see [Binarized MNIST](https://www.tensorflow.org/datasets/catalog/binarized_mnist)). Images are thus instances of *categorical random variables*, sampled from a discrete distribution $P\_\text{data}(\mathbf{x})$. But normalized flows require the use of continuous random variables. If we treat these integers as real numbers[^1] and try to model them with a Normalizing Flow, then we can trivially get arbitrarily high log-likelihoods, even on the test set: let the model concentrate all its probability mass about each of the categorical values. In the limit, this becomes[^2] (in 1-dimension):
$$p\_{\text{model}}(x) =\frac{1}{256} \sum^{255}_{n=0} \delta(x  - n)$$ 
where $\delta(x)$ is the [Dirac delta function](https://en.wikipedia.org/wiki/Dirac_delta_function). This clearly has infinite log-likelihood for any possible value of $x\in \\{0,\ldots,255 \\}$ as $p\_{\text{model}}(x)=\delta(0)/256 = \infty$, and $\log(\infty)=\infty$. I have confirmed this divergence in practice as well: if you try train on MNIST directly, the log-likelihood loss quickly diverges and PyTorch reports NaNs.

One [simple solution](https://arxiv.org/pdf/1511.01844.pdf) is to create a new *dequantized* data distribution $p\_\text{data}(\mathbf{x})$ from the discrete distribution $P\_\text{data}(\mathbf{x})$ by adding uniform noise sampled from $\mathbf{u}\in [0,1]^D$, $\mathbf{y}=\mathbf{x}+\mathbf{u}$. In 1-dimension and for only 4 categorical values, this looks like:

{{< figure library="true" src="2020_flows/quantization.png" title="" lightbox="true" >}}

We can now maximize the new objective $\mathbb{E}\_{\mathbf{y}\sim p\_\text{data}} \left [\log p\_\text{model} (\mathbf{y})\right]$. But this has changed the optimization problem from what we really care about, maximizing the expected log-likelihood of a discrete model on samples generated by the discrete data distribution, $\mathbb{E}\_{\mathbf{x}\sim P\_\text{data}} \left [\log P\_\text{model} (\mathbf{x})\right]$. How are these two optimization problems related? [Theis et al. (2016)](https://arxiv.org/abs/1511.01844) showed that optimizing our new objective is equivalent to maximizing a lower-bound of the objective we actually care about:

$$\mathbb{E}\_{\mathbf{y}\sim p_\text{data}} \left [\log p\_\text{model} (\mathbf{y})\right] \leq
\mathbb{E}\_{\mathbf{x}\sim P\_\text{data}} \left [\log P\_\text{model} (\mathbf{x})\right]
$$

Let us give the derivation here for the 1-dimensional case to simplify things, and consider the MNIST case of $x \in \\{0,\ldots,255\\}$:
\begin{equation}
\begin{aligned}
\mathbb{E}\_{y\sim p\_\text{data}} \left [\log p\_\text{model} (y)\right] & \equiv 
    \int\_{-\infty}^{\infty}p_\text{data}(y) \log p_\text{model}(y) dy \\\\
    & = \sum\_{x=0}^{255} P\_\text{data}(x) \int_0^1 \log p_{\text{model}}(x + u) du \\\\
    & \leq \sum\_{x=0}^{255} P\_\text{data}(x) \log \int_0^1 p_{\text{model}}(x + u) du \\\\
    & = \sum\_{x=0}^{255} P\_\text{data}(x) \log P_\text{model}(x) \\\\
    & \equiv \mathbb{E}\_{y\sim P\_\text{data}} \left [\log P\_\text{model} (y)\right] 
\end{aligned}
\end{equation}
where we used [Jensen's Inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) in the third line, which can be used due to the convexity of the logarithm.

This result also shows that the log-likelihood of our new objective cannot get arbitrarily large, as it is bounded above by the log-likelihood of a discrete model.

A more sophisticated procedure for performing the dequantization can be found in [Ho et al. (2019)](https://arxiv.org/abs/1902.00275).

### Out-of-Distribution Detection

https://arxiv.org/pdf/1810.09136.pdf


####








<!-- Footnotes -->

[^1]: Image pixels are not just any categorical data, but [ordinal categorical data](https://en.wikipedia.org/wiki/Ordinal_data). If it wasn't ordinal, treating it as real continuous data would make no sense at all.
[^2]: This is also known as a [Dirac comb](https://en.wikipedia.org/wiki/Dirac_comb).
[^3]: The mode of an $N$-dimensional multivariate Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$ is at $\mathbf{0}$, and its PDF has value $(2\pi)^{-N/2}$. This is a strict upper-bound for the PDF at any other point. For MNIST with a feature size of $28\times 28 = 784$, the PDF at all points is upper-bounded by $(2\pi)^{-784/2} \approx 10 ^{-313}$. This will certainly cause numerical instabilities if this is our loss value!
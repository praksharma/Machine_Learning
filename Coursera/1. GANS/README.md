# Generative Adversarial Networks (GANs)
## Two network fight with each other
* generator network -> generates Fake images
* discriminator -> Learns to distinguish between fake and real image

Both network learn to do their job during the training. That is is start generator generates rubbish image and discriminator isn't mature enough to identify between real and fake image.

## CNN or FFNN
Generally, GANs are used with CNNs for images, but one can use FFNN too. There are fundamental with FFNN such as Loss of Spatial Information, Spectral Bias, Computational Intensity, Difficulty in Training, overfitting etc.

## Activation function
Since GANs are unsupervised classification model i.e. there are no labels. So, we provide objective to the optimiser similar to a typical unsupervised learning model. The output of the neural network is the probability of image being real or fake using a variation of cross entropy. Thus we use Sigmoid.

## Log-loss functions
It is a minimax game between discriminator and generator. The discriminator is trying to to maximize its probability of correctly classifying real and generated samples, and the generator tries to minimize this probability.

Now if the discriminator is perfect/ pre-trained then, we have insufficient gradient for the generator to catchup. That is why instead of just a maximisation we have minimax objective function.

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
$$

Here, $\mathbb{E}$ is the expectation, $\mathbf{x}$ is the real data, $\mathbf{z}$ is the noise samples drawn from the noise distribution $p_{\mathbf{z}}$.

There are multiple variations of this loss function. Let us break down this loss function:

#### Real loss: $\log D(\mathbf{x})$
* For perfect discriminator the probability for real images $D(\mathbf{x})=1$.
* So, $\log D(\mathbf{x}) = \log (1) = 0$
* In practice, a discriminator that is too confident early in training can cause instability.
* Thus we train Discriminator and it learns alongwith the Generator.

#### Fake loss: $\log(1 - D(G(\mathbf{z})))$
* For perfect discriminator the probability for fake images $D(\mathbf{x})=0$.
* So, $\log(1 - D(G(\mathbf{z}))) = 0$
* Similar to the real loss term, having the discriminator be too confident too early can cause problems, such as vanishing gradients, making it hard for the generator to learn.

So, for a perfect discriminator the loss should be zero.
$$\LARGE \text{Objective: A perfect discriminator}$$ 
## Training
* Use `torch.dataloader` to generate mini-batches.
* Number of batches for both real image and random noise should be same. Although earlier attempts were about ideal number of G updates per D updates. But with GANs, it is pretty much a standard to update both at the same time.
* **First** update the discriminator with the real image and the real loss, then fake image and fake loss.
* **Then** update the generator with fake image and fake loss only.
* In summary the generator learns to produce fake images as close as possible to the real image but it doesn't see the real image directly.





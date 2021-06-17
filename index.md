## Intro

Knowledge transferability, or transfer learning, has been widely adopted to allow a pre-trained model in the source domain to be effectively adapted to downstream tasks in the target domain. It is thus important to explore and understand the factors affecting knowledge transferability. In this paper, as the first work, we analyze and demonstrate the connections between knowledge transferability and another important phenomenon adversarial transferability, i.e., adversarial examples generated against one model can be transferred to attack other models. Our theoretical studies show that adversarial transferability indicates knowledge transferability, and vice versa. Moreover, based on the theoretical
insights, we propose two practical adversarial transferability metrics to characterize this process, serving as bidirectional indicators between adversarial and knowledge transferability. We conduct extensive experiments for different scenarios on diverse datasets, showing a positive correlation between adversarial transferability and knowledge
transferability. Our findings will shed light on future research about effective knowledge transfer learning and adversarial transferability analyses.

![GitHub Logo](/demos/fig1.png)

## Theory

We approach the problem by first setting up the relation between adversarial transferability and gradient matching distance. A key observation is that the adversarial attack reveals the singular vector corresponding to the largest singular value of the Jacobian of function f. Next, we explore the connection to knowledge transferability, which shows the gradient matching distance approximates the function matching distance with a distribution shift
up to a Wasserstein distance. Finally, we complete the analysis by outlining the connection between
the function matching distance and the knowledge transfer loss.

Check out our paper for the detailed derivation at https://arxiv.org/abs/2006.14512 

## Experiment

### Adversarial Transferability Indicates Knowledge-transfer
We control the adversarial transferability by varying the model architectures and measure the knowledge transferability from CIFAR10 to STL-10. The source models include various architecture such as MLP, LeNet, AlexNet and ResNet.
>![GitHub Logo](/demos/fig2.png)

### Knowledge Transferability Indicates Adversarial Transferability
We manually construct five source datasets (5 source models) based on CIFAR-10 and
a single target dataset (1 reference model) based on STL-10. We divide the classes of the original
datasets into two categories, animals (bird, cat, deer, dog) and transportation vehicles (airplane,
automobile, ship, truck). Each of the source datasets consists of different a percentage of animals and
transportation vehicles, while the target dataset contains only transportation vehicles, which is meant
to control the closeness of the two data distributions.
>![GitHub Logo](/demos/fig3.png)

 Check out our paper for more experiments on NLP at https://arxiv.org/abs/2006.14512



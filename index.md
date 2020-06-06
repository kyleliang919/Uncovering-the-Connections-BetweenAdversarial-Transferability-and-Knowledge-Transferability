## Intro

Despite the immense success that deep neural networks (DNNs) have achieved, adversarial examples, which are perturbed inputs that aim to mislead DNNs to make mistakes, have recently led to great concern. On the other hand, adversarial examples exhibit interesting phenomena, such as adversarial transferability. DNNs also exhibit knowledge transfer, which is critical to improving learning efficiency and learning in domains that lack high-quality training data. In this paper, we aim to turn the existence and pervasiveness of adversarial examples into an advantage. Given that adversarial transferability is easy to measure while it can be challenging to estimate the effectiveness of knowledge transfer, does adversarial transferability indicate knowledge transferability? We first theoretically analyze the relationship between adversarial transferability and knowledge transferability and outline easily checkable sufficient conditions that identify when adversarial transferability indicates knowledge transferability. In particular, we show that composition with an affine function is sufficient to reduce the difference between two models when adversarial transferability between them is high. We provide empirical evaluation for different transfer learning scenarios on diverse datasets, including CIFAR-10, STL-10, CelebA, and Taskonomy-data -- showing a strong positive correlation between the adversarial transferability and knowledge transferability, thus illustrating that our theoretical insights are predictive of practice.
![GitHub Logo](/demos/fig1.png)

## Theoretical results

> Intuitive low dimensional visualization of Adversarial Transferability: 
![GitHub Logo](/demos/fig2.png)

![GitHub Logo](/demos/notation.png)
![GitHub Logo](/demos/setting.png)
![GitHub Logo](/demos/problem.png)
![GitHub Logo](/demos/def1.png)
![GitHub Logo](/demos/def2.png)
![GitHub Logo](/demos/def3.png)
![GitHub Logo](/demos/th1.png)
![GitHub Logo](/demos/th2.png)
![GitHub Logo](/demos/th3.png)

## Experiment results

### Adversarial Transferability Indicates Knowledge-transfer among Data Distributions
We manually construct five source datasets (5 source models) based on CIFAR-10 and
a single target dataset (1 reference model) based on STL-10. We divide the classes of the original
datasets into two categories, animals (bird, cat, deer, dog) and transportation vehicles (airplane,
automobile, ship, truck). Each of the source datasets consists of different a percentage of animals and
transportation vehicles, while the target dataset contains only transportation vehicles, which is meant
to control the closeness of the two data distributions.
![GitHub Logo](/demos/fig3.png)
### Adversarial Transferability Indicating Knowledge-transfer among Attributes
In addition to the data distributions, we validate our theory on another dimension, attributes. This experi-
ment suggests that the more adversarially transferable the source model of certain attributes is to the reference
model, the better the model performs on the target task aiming to learn tar- get attributes.
CelebA consists of 202,599 face images from 10,177 identities. A reference facial recognition model is trained on this identities. Each image also comes with 40 binary attributes, on which we train 40 source models. Our goal is to test
whether source models of source attributes, can transfer to perform facial recognition. Below we show the top-5 attributes that have the highest adversarial transferability.
![GitHub Logo](/demos/fig4.png)
### Adversarial Transferability Indicating Knowledge-transfer among Tasks
![GitHub Logo](/demos/fig5.png)


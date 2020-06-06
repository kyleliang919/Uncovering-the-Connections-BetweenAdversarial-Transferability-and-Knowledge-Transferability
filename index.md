## Intro

Despite the immense success that deep neural networks (DNNs) have achieved, adversarial examples, which are perturbed inputs that aim to mislead DNNs to make mistakes, have recently led to great concern. On the other hand, adversarial examples exhibit interesting phenomena, such as adversarial transferability. DNNs also exhibit knowledge transfer, which is critical to improving learning efficiency and learning in domains that lack high-quality training data. In this paper, we aim to turn the existence and pervasiveness of adversarial examples into an advantage. Given that adversarial transferability is easy to measure while it can be challenging to estimate the effectiveness of knowledge transfer, does adversarial transferability indicate knowledge transferability? We first theoretically analyze the relationship between adversarial transferability and knowledge transferability and outline easily checkable sufficient conditions that identify when adversarial transferability indicates knowledge transferability. In particular, we show that composition with an affine function is sufficient to reduce the difference between two models when adversarial transferability between them is high. We provide empirical evaluation for different transfer learning scenarios on diverse datasets, including CIFAR-10, STL-10, CelebA, and Taskonomy-data -- showing a strong positive correlation between the adversarial transferability and knowledge transferability, thus illustrating that our theoretical insights are predictive of practice.

![GitHub Logo](/demos/fig1.png)
## Theoretical Results
Intuitive low dimensional visualization of adversarial transferability: ![GitHub Logo](/demos/fig2.png)
![GitHub Logo](/demos/notation.png)
![GitHub Logo](/demos/setting.png)
![GitHub Logo](/demos/problem.png)
![GitHub Logo](/demos/def1.png)
![GitHub Logo](/demos/def2.png)
![GitHub Logo](/demos/def3.png)
![GitHub Logo](/demos/th1.png)
![GitHub Logo](/demos/th2.png)
![GitHub Logo](/demos/th3.png)

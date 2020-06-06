## Abstract

Despite the immense success that deep neural networks (DNNs) have achieved, \emph{adversarial examples}, which are perturbed inputs that aim to mislead DNNs to make mistakes, have recently led to great concern. On the other hand, adversarial examples exhibit interesting phenomena, such as \emph{adversarial transferability}. DNNs also exhibit knowledge transfer, which is critical to improving learning efficiency and learning in domains that lack high-quality training data. In this paper, we aim to turn the existence and pervasiveness of adversarial examples into an advantage. Given that adversarial transferability is easy to measure while it can be challenging to estimate the effectiveness of knowledge transfer, \emph{does adversarial transferability indicate knowledge transferability?} We first theoretically analyze the relationship between adversarial transferability and knowledge transferability and outline easily checkable sufficient conditions that identify when adversarial transferability indicates knowledge transferability. In particular, we show that composition with an affine function is sufficient to reduce the difference between two models when adversarial transferability between them is high. We provide empirical evaluation for different transfer learning scenarios on diverse datasets, including CIFAR-10, STL-10, CelebA, and Taskonomy-data -- showing a strong positive correlation between the adversarial transferability and knowledge transferability, thus illustrating that our theoretical insights are predictive of practice.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/AI-secure/Does-Adversairal-Transferability-Indicate-Knowledge-Transferability/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

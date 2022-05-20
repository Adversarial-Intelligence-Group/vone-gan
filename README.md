<h1 align="center">Generative adversarial networks with bio-inspired primary visual cortex for Industry 4.0</h1>

<h4 align="center">Vladyslav Branytskyi, Mariia Golovianko, Diana Malyk, Vagan Terziyan</h4>

<p align="center"> [<b><a href="https://doi.org/10.1016/j.procs.2022.01.240">Paper</a></b>] &emsp; [<b><a href="#citation">Citation</a></b>] </p>

Biologically-inspired Generative Neural Networks (GANs). VOneNets have the following features:
- Fixed-weight neural network model of the primate primary visual cortex (V1) as the front-end
- Robust to image perturbations
- Brain-mapped
- Flexible: can be adapted to different Discriminator architectures

## Setup and training a model
1. You need to clone it in your local repository
  `$ git clone https://github.com/Adversarial-Intelligence-Group/vone-gan.git`
3. Install dependencies using pipenv
4. Run the code
  `$ python3 train.py`

## Dataset
Conveyor Belt dataset can be found at [this link](https://drive.google.com/drive/folders/1TycfdqGih3hHYS3hM70rybKH8rfGUiyr?usp=sharing).

## Citation

If you use the results presented in this paper or the code from the repository, please cite the relevant [paper]():
```
@article{BRANYTSKYI2022418,
  title = {Generative adversarial networks with bio-inspired primary visual cortex for Industry 4.0},
  journal = {Procedia Computer Science},
  volume = {200},
  pages = {418-427},
  year = {2022},
  note = {3rd International Conference on Industry 4.0 and Smart Manufacturing},
  issn = {1877-0509},
  doi = {https://doi.org/10.1016/j.procs.2022.01.240},
  url = {https://www.sciencedirect.com/science/article/pii/S1877050922002496},
  author = {Vladyslav Branytskyi and Mariia Golovianko and Diana Malyk and Vagan Terziyan},
  keywords = {Biologicalization, Industry 4.0, GAN, VOneGAN, primary visual cortex V1, hybrid CNN},
  abstract = {Biologicalization (biological transformation) is an emerging trend in Industry 4.0 affecting digitization of manufacturing and related        processes. It brings up the next generation of manufacturing technology and systems that extensively use biological and bio-inspired principles, materials, functions, structures and resources. This research is a contribution to the further convergence of computer and human vision for more robust and accurate automated object recognition and image generation. We present VOneGANs, a novel class of generative adversarial networks (GANs) with the qualitatively updated discriminative component. The new model incorporates a biologically constrained digital primary visual cortex V1. This earliest cortical visual area performs the first stage of human‘s visual processing and is believed to be a reason of its robustness and accuracy. Experiments with the updated architectures confirm the improved stability of GANs training and the higher quality of the automatically generated visual content. The promising results allow considering VOneGANs as providers of high-quality training content and as enablers of future simulation-based decision-making and decision-support tools for condition-monitoring, supervisory control, diagnostics, predictive maintenance, and cybersecurity in Industry 4.0.}
}
```

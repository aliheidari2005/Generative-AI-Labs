# Generative-AI-Labs

Hands-on implementations of **flow matching** and **diffusion models** from scratch in PyTorch, following the MIT *Flow Matching and Diffusion Models* lab series. The labs go from simulating ODEs/SDEs on toy data to training a Diffusion Transformer (DiT) for conditional image generation on MNIST.

## Contents

| Notebook | Topic |
|---|---|
| `lab_one.ipynb` | **Lab 1 – Flow & diffusion foundations.** Vector fields, ODE/SDE simulation (Euler, Euler–Maruyama), conditional probability paths, and sampling from them. |
| `lab_two.ipynb` | **Lab 2 – Training generative models.** Conditional flow matching and score matching objectives, training on 2D toy distributions, and comparing ODE vs. SDE sampling. |
| `lab_three.ipynb` | **Lab 3 – Conditional image generation.** Class-conditional generation on MNIST with classifier-free guidance (CFG), using a Diffusion Transformer (DiT) as the denoising/velocity network. |

```
Generative-AI-Labs/
├── data/MNIST/raw/     # MNIST dataset (downloaded by torchvision)
├── lab_one.ipynb
├── lab_two.ipynb
├── lab_three.ipynb
├── .gitignore
└── README.md
```

## Key Ideas Covered

- **Flow models:** learning a time-dependent vector field that transports noise to data via an ODE.
- **Diffusion models:** adding stochasticity through an SDE and learning the score function.
- **Conditional flow matching:** a simple regression objective on conditional probability paths.
- **Classifier-free guidance:** jointly training conditional and unconditional models, then interpolating at sampling time to control sample fidelity.
- **Diffusion Transformer (DiT):** transformer-based architecture with time/class conditioning for image generation.

## Getting Started

### Requirements

- Python 3.9+
- PyTorch and torchvision
- NumPy, Matplotlib
- Jupyter Notebook or JupyterLab (or Google Colab)

```bash
git clone https://github.com/aliheidari2005/Generative-AI-Labs.git
cd Generative-AI-Labs
pip install torch torchvision numpy matplotlib jupyter
jupyter notebook
```

Then open the notebooks in order (`lab_one` → `lab_two` → `lab_three`). A GPU is recommended for Lab 3 but not required for Labs 1–2.

## Results

> Add sample outputs here, e.g. generated MNIST digits for different guidance scales:
>
> `![MNIST samples](images/mnist_samples.png)`

## References

- Lipman et al., *Flow Matching for Generative Modeling*, 2023
- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, 2021
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, 2022
- Peebles & Xie, *Scalable Diffusion Models with Transformers (DiT)*, 2023
- MIT course *Flow Matching and Diffusion Models* (labs and lecture notes)

## Author

**Ali Heidari** – [@aliheidari2005](https://github.com/aliheidari2005)

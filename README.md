# Forget More to Learn More: A PyTorch Implementation

This repository contains a PyTorch implementation of the paper:\
**"Forget More to Learn More: Domain-specific Feature Unlearning for
Semi-supervised and Unsupervised Domain Adaptation"**\
by *Hritam Basak and Zhaozheng Yin*.

The project implements the novel episodic learning strategy of *"learn,
forget, then learn more"* for Semi-supervised Domain Adaptation (SSDA).\
The core idea is to first explicitly learn domain-specific features,
then unlearn (forget) them to create a domain-agnostic representation,
and finally learn a robust classifier on this cleaned representation.

------------------------------------------------------------------------

## Methodology Overview

The training process is divided into two main stages:

### Stage 1: Learn Domain-specific Features

In this stage, two separate encoder-classifier pairs are trained:

-   **Source Pair (E_S, C_S):** Trained to classify images from the
    source domain correctly. To ensure it only learns source-specific
    features, it's also trained to be uncertain about images from the
    target domain using an entropy maximization loss.

-   **Target Pair (E_T, C_T):** Similarly trained on the few available
    labeled target samples. It is simultaneously trained to be uncertain
    about source domain images.

This adversarial process results in two expert models, each specialized
in its own domain's features.

### Stage 2: Forget and Learn More

This stage trains a **ReconstructionNetwork (R)** and a final
**DomainAgnosticClassifier (C)**.

#### Forget (Unlearn):

-   Images from both domains are passed through the
    ReconstructionNetwork.
-   The core "unlearning" happens by feeding these reconstructed images
    into their respective frozen, domain-specific models (E_S, C_S and
    E_T, C_T).
-   An **unlearning loss (entropy maximization)** is applied, forcing
    the ReconstructionNetwork to generate images that the
    domain-specific classifiers cannot recognize.\
    This effectively strips out the domain-specific features.
-   A **reconstruction loss (MSELoss)** ensures the reconstructed images
    still retain essential structural information from the originals.

#### Learn More:

-   **Gaussian-guided Latent Alignment (GLA):** The latent features from
    the reconstructed images are aligned to a standard Gaussian prior
    using a KL-Divergence loss.\
    This pushes both domains into a shared, domain-agnostic feature
    space.
-   **Supervised Learning:** The final DomainAgnosticClassifier is
    trained on the domain-agnostic features from the reconstructed
    source images and labeled target images.

At inference time, a target image is passed through the trained
ReconstructionNetwork, the frozen encoder_T, and finally the
DomainAgnosticClassifier to get a prediction.

------------------------------------------------------------------------

## File Structure

    .
    ├── main.py                # Main script to run the two-stage training process.
    ├── models.py              # Contains definitions for Encoder, Classifier, and ReconstructionNetwork (U-Net).
    ├── data_loader.py         # Handles loading and splitting of source and target domain data.
    ├── utils.py               # Helper functions, including EntropyLoss and KL-Divergence loss.
    └── README.md              # This file.

------------------------------------------------------------------------

## Setup and Installation

### 1. Prerequisites

-   Python 3.8+\
-   PyTorch 1.10+\
-   torchvision\
-   tqdm\
-   numpy

### 2. Installation

Clone the repository and install the required packages:

``` bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt   # Or install manually
pip install torch torchvision tqdm numpy
```

### 3. Dataset

Download a domain adaptation dataset like **Office-Home**.\
You must structure the dataset folders by domain name as follows:

    ./data/OfficeHome/
    ├── Art/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    ├── Clipart/
    │   ├── ...
    ├── Product/
    │   └── ...
    └── Real_World/
        └── ...

------------------------------------------------------------------------

## How to Run

You can run the entire two-stage training process using `main.py`.\
You can specify the source and target domains, data path, and other
hyperparameters via command-line arguments.

**Example:**\
To run an adaptation experiment from the 'Art' domain to the 'Clipart'
domain on the Office-Home dataset:

``` bash
python main.py     --data_path ./data/OfficeHome     --source Art     --target Clipart     --num_classes 65     --epochs_stage1 10     --epochs_stage2 20     --batch_size 16
```

------------------------------------------------------------------------

## Command-Line Arguments

-   `--data_path`: Path to the root of the dataset (e.g.,
    ./data/OfficeHome).\
-   `--source`: Name of the source domain folder.\
-   `--target`: Name of the target domain folder.\
-   `--num_classes`: Number of classes in the dataset (65 for
    Office-Home).\
-   `--batch_size`: Batch size for training.\
-   `--lr`: Learning rate. Default: 1e-4.\
-   `--weight_decay`: Weight decay for the AdamW optimizer. Default:
    5e-2.\
-   `--epochs_stage1`: Number of epochs for Stage 1. Default: 10.\
-   `--epochs_stage2`: Number of epochs for Stage 2. Default: 20.\
-   `--num_labeled`: Number of labeled samples per class in the target
    domain for SSDA. Default: 3.\
-   `--alpha`: Weight for uncertainty loss in Stage 1. Default: 1.0.\
-   `--beta`: Weight for reconstruction loss in Stage 2. Default: 10.0.\
-   `--gamma`: Weight for unlearning loss in Stage 2. Default: 0.1.\
-   `--delta`: Weight for KL divergence loss in Stage 2. Default: 1.0.

------------------------------------------------------------------------

## Citation

If you use this work, please consider citing the original paper:

``` bibtex
@article{basak2023forget,
  title={Forget More to Learn More: Domain-specific Feature Unlearning for Semi-supervised and Unsupervised Domain Adaptation},
  author={Basak, Hritam and Yin, Zhaozheng},
  journal={arXiv preprint arXiv:2307.05513},
  year={2023}
}
```

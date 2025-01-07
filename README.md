# Uncertainty-Aware Regularization for Image-to-Image Translation - WACV 2025

## Abstract

The importance of quantifying uncertainty in deep networks has become paramount for reliable real-world applications. In this paper, we propose a method to improve uncertainty estimation in medical Image-to-Image (I2I) translation. Our model integrates aleatoric uncertainty and employs Uncertainty-Aware Regularization (UAR) inspired by simple priors to refine uncertainty estimates and enhance reconstruction quality. 

We show that by leveraging simple priors on parameters, our approach captures more robust uncertainty maps, effectively refining them to indicate precisely where the network encounters difficulties, while being less affected by noise. Our experiments demonstrate that UAR not only improves translation performance but also provides better uncertainty estimations, particularly in the presence of noise and artifacts. 

We validate our approach using two medical imaging datasets, showcasing its effectiveness in maintaining high confidence in familiar regions while accurately identifying areas of uncertainty in novel or ambiguous scenarios.

---

## Block Diagram

![Block Diagram](https://github.com/anuja13/Uncertainity-Aware-Regularizatopn--UAR-/blob/main/blockdiag.1.pdf "Block Diagram")

**Figure**: Comparison between Probabilistic I2I translation, Non-Probabilistic I2I translation, and Uncertainty-Aware Regularization (ours). Residuals in our method are modeled as distributions, with parameters regularized for better uncertainty estimation and performance.

---

## Citation

If you find this work useful, please cite our paper as follows:

```bibtex
@inproceedings{vats2025uar,
  title     = {Uncertainty-Aware Regularization for Image-to-Image Translation},
  author    = {Anuja Vats, Ivar Farup, Marius Pedersen, Kiran Raja},
  booktitle = {Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025}
}

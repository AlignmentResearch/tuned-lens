# Tuned Lens 🔎
<a target="_blank" href="https://colab.research.google.com/github/AlignmentResearch/tuned-lens/blob/main/notebooks/interactive.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://huggingface.co/spaces/AlignmentResearch/tuned-lens">
<img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg", alt="Open in Spaces">
</a>


Tools for understanding how transformer predictions are built layer-by-layer

<img src=https://user-images.githubusercontent.com/12176390/224879115-8bc95f26-68e4-4f43-9b4c-06ca5934a29d.png>

This package provides a simple interface training and evaluating __tuned lenses__. A tuned lens allows us to peak at the iterative computations a transformer uses to compute the next token.


## What is a Lens?
<img alt="A diagram showing how a translator within the lens allows you to skip intermediate layers." src="https://user-images.githubusercontent.com/12176390/227057947-1ef56811-f91f-48ff-8d2d-ff04cc599125.png"  width=400\>

A lens into a transformer with $n$ layers allows you to replace the last $m$ layers of the model with an [affine transformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (we call these affine translators). Each affine translator, is trained to minimize the KL divergence between its prediction and the final output distribution of the original model. This means that after training the tuned lens allows you to skips over these last few layers and see the best prediction that can be made from the model's intermediate representations, i.e. the residual stream, at layer $n - m$.

The reason we need to train a affine translator is because the representations may be rotated, shifted, or stretched from layer to layer. This training is what differentiates this method from simpler approaches that unmbed the residual stream of the network directly using the unembeding matrix i.e. the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). We explain this process and its applications in the paper [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112).

### Acknowledgments
Originally concieved by [Igor Ostrovsky](https://twitter.com/igoro) and [Stella Biderman](https://www.stellabiderman.com/) at [EleutherAI](https://www.eleuther.ai/), this library was built as a collaboration between FAR and EleutherAI researchers.

> **Warning**
> This package has not reached 1.0 yet. Expect the public interface to change regularly and without a major version bump.

## Install instructions
### Installing From Source
First you will need to install the basic prerequisites into a virtual environment
* Python 3.9+
* Pytorch 1.12.0+

then you can simply install the package using pip.
```
pip install tuned-lens
```

### Install Using Docker
If you prefer to run the code from within a container you can use the provided docker
file
```
docker pull ghcr.io/alignmentresearch/tuned-lens:latest
docker run --rm tuned-lens:latest tuned-lens --help
```

## Contributing
Make sure to install the dev dependencies and install the pre-commit hooks
```
$ git clone https://github.com/AlignmentResearch/tuned-lens.git
$ pip install -e ".[dev]"
$ pre-commit install
```

## Citation

If you find this library useful, please cite it as

```bibtex
@article{belrose2023eliciting,
  title={Eliciting Latent Predictions from Transformers with the Tuned Lens},
  authors={Belrose, Nora and Furman, Zach and Smith, Logan and Halawi, Danny and McKinney, Lev and Ostrovsky, Igor and Biderman, Stella and Steinhardt, Jacob},
  journal={to appear},
  year={2023}
}
```

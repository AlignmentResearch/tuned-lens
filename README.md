# Tuned Lens 🔎
Tools for understanding how transformer predictions are built layer-by-layer

![Using the Tuned-lens](https://user-images.githubusercontent.com/12176390/224879115-8bc95f26-68e4-4f43-9b4c-06ca5934a29d.png)

This package provides a simple interface training and evaluating __tuned lenses__. A tuned lens allows us to peak at the iterative computations that a transformer is using the compute the next token.

A lens into a transformer with n layers allows you to replace the last $m$ layers of the model with an [affine transformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (we call these affine translators).

This skips over these last few layers and lets you see the best prediction that can be made from the model's intermediate representations, i.e. the residual stream, at layer $n - m$. Since the representations may be rotated, shifted, or stretched from layer to layer it's useful to train an affine specifically on each layer. This training is what differentiates this method from simpler approaches that decode the residual stream of the network directly using the unembeding layer i.e. the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). We explain this process and its applications in a forthcoming paper "Eliciting Latent Predictions from Transformers with the Tuned Lens".

### Acknowledgments
Originally concieved by [Igor Ostrovsky](https://twitter.com/igoro?lang=en) and [Stella Biderman](www.stellabiderman) at [EleutherAI](www.eleuther.ai), this library was built as a collaboration between FAR and EleutherAI researchers.

> **Warning**
> This package has not reached 1.0 yet. Expect the public interface to change regularly and without a major version bump.

## Install instructions
### Installing From Source
First you will need to install the basic prerequisites into a virtual environment
* Python 3.9+
* Pytorch 1.12.0+

then you can simply install the package using pip.
```
git clone https://github.com/AlignmentResearch/tuned-lens
cd tuned-lens
pip install .
```

### Install Using Docker
If you prefer to run the code from within a container you can use the provided docker
file
```
git clone https://github.com/AlignmentResearch/tuned-lens
cd tuned-lens
docker build -t tuned-lens-prod --target prod .
```

## Quick Start Guid
### Downloading the datasets
```
wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
unzstd val.jsonl.zst

wget https://the-eye.eu/public/AI/pile/test.jsonl.zst
unzstd test.jsonl.zst
```

### Evaluating a Lens
Once you have a lens file either by training it yourself of by downloading it. You
can run various evaluations on it using the provided evaluation command.
```
tuned-lens eval gpt2 test.jsonl --lens gpt-2-lens
    --dataset the_pile all \
    --split validation \
    --output lens_eval_results.json
```


### Training a Lens
This will train a tuned lens on gpt-2 with the default hyper parameters.

```bash
tuned-lens train gpt2 val.jsonl
    --dataset the_pile all \
    --split validation \
    --output gpt-2-lens
```

> **Note**
> This will download the entire validation set of the pile which is over 30 GBs. If you
> are doing this within a docker file it's recommended to mount external storage to huggingface's
> cache directory.

## Contributing
Make sure to install the dev dependencies and install the pre-commit hooks
```
$ pip install -e ".[dev]"
$ pre-commit install
```


## Citation Information

If you find this library useful, please cite it as

```bibtex
@article{belrose2023eliciting,
  title={Eliciting Latent Predictions from Transformers with the Tuned Lens},
  authors={Belrose, Nora and Furman, Zach and Smith, Logan and Halawi, Danny and McKinney, Lev and Ostrovsky, Igor and Biderman, Stella and Steinhardt, Jacob},
  journal={to appear},
  year={2023}
}
```

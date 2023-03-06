# Tuned Lens 🔎
Tools for understanding how transformer predictions are built layer-by-layer

<img width="1028" alt="tuned lens" src="https://user-images.githubusercontent.com/39116809/206883419-4fb9083d-3fa0-48e9-ba97-b70cb21b08e9.png">

This package provides a simple interface training and evaluating __tuned lenses__. A tuned lens allows us to peak at the iterative computations that a transformer is using the compute the next token.

A lens into a transformer with n layers allows you to replace the last $m$ layers of the model with an [affine transformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (we call these affine adapters).

This essentially skips over these last few layers and lets you see the best prediction that can be made from the model's representations, i.e. the residual stream, at layer $n - m$. Since the representations may be rotated, shifted, or stretched from layer to layer it's useful to train the len's affine adapters specifically on each layer. This training is what differentiates this method from simpler approaches that decode the residual stream of the network directly using the unembeding layer i.e. the logit lens. We explain this process in more detail in a forthcoming paper.

## Install instructions
### Installing From Source
First you will need to install the basic prequists into a virtual envirment
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
tuned-lens eval gpt-2 test.jsonl --lens gpt-2-lens 
    --dataset the_pile all \
    --split validation \
    --output lens_eval_results.json
```


### Training a Lens
This will train a tuned lens on gpt-2 with the default hyper parameters.

```bash
tuned-lens train gpt-2 val.jsonl
    --dataset the_pile all \
    --split validation \
    --output gpt-2-lens
```

> **Note**
> This will download the entire validation set of the pile which is over 30 GBs. If you
> are doing this within a docker file it's recomended to mount external storage to huggingface's
> cache directory.

### Citation

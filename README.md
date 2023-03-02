# Tuned Lens 🔎
Tools for understanding how transformer predictions are built layer-by-layer

<img width="1028" alt="tuned lens" src="https://user-images.githubusercontent.com/39116809/206883419-4fb9083d-3fa0-48e9-ba97-b70cb21b08e9.png">

This package provides a simple interface training and evaluating __tuned lenses__. A lens
into a tranformer with n layers allows you to replace the last $m$ layers of the model with an 
[affine tranformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (we call these affine adapters).
This essentaily skips over these last few layers and lets you to see the best prediction that can be made from the model's representations, i.e. the residual stream,
at layer $n - m$. Since the representations may be rotated, shifted, or streched from layer to layer it's useful to train the len's affine adapters specificaly on
each layer. This training is what diffentaits this method from simpleir approches that decode the residual stream of the network directly using the unembeding layer i.e. the logit lens. We explain this process in more detail in a forth coming paper.


## Install instructions
### Installing from source
First you will need to install the basic prequists into a virtual envirment
* Python 3.9+
* Pytorch 1.12.0+

then you can simply install the package using pip.
```
git clone https://github.com/norabelrose/tuned-lens
cd tuned-lens
pip install .
```

### Install using docker
If you perfer to run the code from within a container you can use the provided docker
file
```
git clone https://github.com/norabelrose/tuned-lens
cd tuned-lens
docker build -t tuned-lens-prod --target prod .
```

## Quick start guid
### Evaluating a lens
Once you have a lens file either by training it yourself of by downloading it. You
can run various evaluations on it using the provided evaluation command.
```
tuned-lens eval gpt-2 --lens gpt-2-lens 
    --dataset the_pile all \
    --split validation \
    --output lens_eval_results.json
```


### Training a lens
This will train a tuned lens on gpt-2 with the default hyper parameters.

```bash
tuned-lens train gpt-2 
    --dataset the_pile all \
    --split validation \
    --output gpt-2-lens
```

> **Note**
> This will download the entire validation set of the pile which is over 30 GBs. If you
> are doing this within a docker file it's recomended to mount external storage to huggingface's
> cache directory.

### Citation

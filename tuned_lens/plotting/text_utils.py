import torch
from IPython.display import display, HTML
from typing import Tuple, List, Optional
from captum.attr import visualization as viz
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from plotly.express.colors import sample_colorscale
from typing import List


def formate_tokens_and_scalers(tokens: List[str], scalers: List[float], color_scale: str = 'reds') -> str:
    assert len(tokens) == len(scalers)
    colors = sample_colorscale('reds', scalers)
    tags = ["<td>"]
    for idx, (token, color) in enumerate(zip(tokens, colors)):
        unwrapped_tag = (
            '<mark style="background-color: {color}; opacity:1.0; line-height:1.75">'
                    '<font color="black"> {token} </font>'
            '</mark>'
        ).format(color=color, token=token, idx=idx)
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def visualize_text(tokens: List[str], scalers: List[float]) -> HTML:
    dom = ["<div>"]
    dom.append(formate_tokens_and_scalers(tokens, scalers))
    dom.append('</div>')
    html = HTML("".join(dom))
    return html

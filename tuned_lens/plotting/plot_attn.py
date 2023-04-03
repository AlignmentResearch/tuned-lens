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


def plot_attention(
    attentions: Tuple[torch.FloatTensor],
    tokens: List[str],
    token_position: int = 0,
    layer: int = 0,
    head: int = 0,
) -> HTML:
    """Produce a norm weighted visualization of the attention

    Args:
        attentions: Tuple of torch.FloatTensor (one for each layer) of shape 
            (batch_size, num_heads, sequence_length, sequence_length).
        past_key_values: A tuple of FloatTensors (one for each layer) of shape 
            (batch_size, num_heads, sequence_length - 1, embed_size_per_head)
        output_matricies: The output tensors from the model
            ()
        tokens: A list of the string representation of each token.
        token_position: Which toke positions attention to visualize.
            Note this can be negative.
        layer: the specific layer to attend plot the attention of
        head: the specific head to plot. If none (default) then average the attention
            scores over the heads.

    Returns:
        A html visualization showing the attention weighted by the norm of the vectors
        being attended to.
    """
    # Select the specific attention tensor, hidden state tensor, and token position
    attention = attentions[layer][0, head]
    token_attentions = attention[token_position]

    # Convert the weighted attention values to a list
    attentions_list = token_attentions.cpu().detach().tolist()

    return visualize_text(tokens, attentions_list)

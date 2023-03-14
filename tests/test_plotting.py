from tuned_lens.nn.lenses import TunedLens, LogitLens
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.plotting import plot_lens
import torch as th


def test_plot_logit_lens():
    pythia_125M_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-125m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-125m")
    pythia_125M_lens = TunedLens(pythia_125M_model)
    pythia_125M_lens.attn_translators = th.nn.ModuleList()
    text = "Never gonna give you up, never gonna let you down,"
    text += " never gonna run around and desert you"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    # plot logit lens
    plot_lens(
        pythia_125M_model,
        lens=LogitLens(pythia_125M_model),
        input_ids=input_ids,
        tokenizer=tokenizer,
    )
    # plot w/ tuned lens
    plot_lens(
        pythia_125M_model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        lens=pythia_125M_lens,
    )
    # plot w/ tuned lens and last layer output
    plot_lens(
        pythia_125M_model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        lens=pythia_125M_lens,
    )

    # Plot w/ short text
    short_text = "Never gonna"
    short_input_ids = tokenizer(short_text, return_tensors="pt").input_ids
    plot_lens(
        pythia_125M_model,
        input_ids=short_input_ids,
        tokenizer=tokenizer,
        lens=pythia_125M_lens,
    )

    # Plot topk prob diff w/ last layer, should have N/A for the bottom row
    plot_lens(
        pythia_125M_model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        lens=pythia_125M_lens,
        topk_diff=True,
    )
    # Plot topk prob diff w/o last layer
    plot_lens(
        pythia_125M_model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        lens=pythia_125M_lens,
        topk_diff=True,
    )

    # Plot w/ topk equals large
    plot_lens(
        pythia_125M_model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        lens=pythia_125M_lens,
        topk=30,
    )  # Breaks at 18 for notebook, but good for browser
    # Plot w/ topk equals small
    plot_lens(
        pythia_125M_model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        lens=pythia_125M_lens,
        topk=1,
    )

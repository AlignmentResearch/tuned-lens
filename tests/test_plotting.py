from white_box.nn.tuned_lens import TunedLens
from transformers import AutoModelForCausalLM, AutoTokenizer
from white_box.plotting import plot_logit_lens


lens_directory = "C:/Users/logan/Documents/GitHub/white-box/Data/pythia/next"
mlp_lens_directory = "C:/Users/logan/Documents/GitHub/white-box/Data/mlp/mlp"
pythia_125M_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-125m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-125m")
pythia_125M_lens = TunedLens.load(lens_directory)
mlp_lens = TunedLens.load(mlp_lens_directory)
text = "Never gonna give you up, never gonna let you down, never gonna run around and desert you"
input_ids = tokenizer(text, return_tensors="pt").input_ids

#plot logit lens 
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, test_text="Logit Lens")
#plot w/ tuned lens
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, tuned_lens = pythia_125M_lens, test_text="Tuned Lens")
#plot w/ tuned lens and last layer output
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, tuned_lens = pythia_125M_lens, add_last_tuned_lens_layer = True, test_text="Tuned Lens + Last Layer")
#plot mlp lens
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, tuned_lens = mlp_lens, test_text="MLP Lens")

#Plot w/ short text
short_text = "Never gonna"
short_input_ids = tokenizer(short_text, return_tensors="pt").input_ids
plot_logit_lens(pythia_125M_model, input_ids=short_input_ids, tokenizer = tokenizer, tuned_lens = pythia_125M_lens, add_last_tuned_lens_layer = True, test_text="Short Text")

#Plot topk prob diff w/ last layer, should have N/A for the bottom row
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, tuned_lens = pythia_125M_lens, add_last_tuned_lens_layer = True, topk_diff = True, test_text="Topk Prob Diff w/ Last Layer. N/A for bottom row")
#Plot topk prob diff w/o last layer
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, tuned_lens = pythia_125M_lens, topk_diff = True, test_text="Topk Prob Diff w/o Last Layer. N/A for bottom row")

#Plot Rank 1
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, rank=1, test_text="Rank 1")
#Plot Rank 2
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, rank=2, test_text="Rank 2")

#Plot w/ topk equals large
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, tuned_lens = pythia_125M_lens, add_last_tuned_lens_layer = True, topk=30, test_text="Large TopK Hovertip")  #Breaks at 18 for notebook, but good for browser
#Plot w/ topk equals small
plot_logit_lens(pythia_125M_model, input_ids=input_ids, tokenizer = tokenizer, tuned_lens = pythia_125M_lens, add_last_tuned_lens_layer = True, topk=1, test_text="Small TopK Hovertip")
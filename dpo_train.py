import torch
from tqdm import tqdm
from trl import DPOTrainer, create_reference_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments

from load_data import load_dpo_data
from utils import get_parse, setup_seed, setLogger

args = get_parse()
args.eval_type = 'drd2_qed_sa'

model_path = "model_weights/checkpoints/gpt2_ft/checkpoint-1360000"

# config = DPOConfig(
#     model_name="model_weights/checkpoints/gpt2_ft/checkpoint-680000",
#     learning_rate=1.41e-5,
#     batch_size=512
#     # log_with="wandb",
# )

model = GPT2LMHeadModel.from_pretrained(model_path)
ref_model = create_reference_model(model)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dpo_data(args, tokenizer)
training_args = TrainingArguments(
    output_dir="./model_weights/checkpoints/gpt2_dpo",
    learning_rate=5e-6,
    report_to="none"
    )

dpo_trainer = DPOTrainer(
    model,
    ref_model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
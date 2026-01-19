"""
Trainer callback module
Inspired mostly from https://github.com/DhananjayAshok/llm-utils/blob/dev/training/trainers.py
"""

import torch
from transformers import TrainerCallback
import wandb


ZERO_LOSS = 1e-2


class StopOnZeroLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Check if the training loss is available and is approximately zero
        if state.log_history:
            last_log = state.log_history[-1]
            for k in ["loss", "train_loss"]:
                if k in last_log and last_log[k] < ZERO_LOSS:
                    print(f"Training loss reached zero ({last_log[k]}), stopping training...")
                    control.should_training_stop = True
        return control


 # TODO: test GRPO too
class SampleLoggingCallback(TrainerCallback):
    def __init__(self, training_kind, eval_max_new_tokens: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_kind = training_kind
        self.eval_max_new_tokens = eval_max_new_tokens
        self.input_ids_key_name = "input_ids"
        self.output_ids_key_name = "labels"        
        if self.training_kind in ["dpo"]:
            self.input_ids_key_name = "prompt_input_ids"
            self.output_ids_key_name = "chosen_input_ids"
            self.rejected_ids_key_name = "rejected_input_ids"
        base_columns = ["global_step", "item_id", "input"]
        if self.training_kind in ["clf", "sft", "pre"]:
            base_columns.extend(["target_output", "model_output"])
        elif self.training_kind in ["dpo"]:
            base_columns.extend(["chosen_output", "rejected_output", "model_output"])
        self.table = wandb.Table(columns=base_columns)

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        batch = next(iter(eval_dataloader))
        all_input_texts = []
        all_targets = [] # also all_chosens
        all_outputs = []
        all_rejecteds = []
        tokenizer = kwargs.get("processing_class")

        if self.training_kind == "clf":
            input_texts = tokenizer.batch_decode(batch[self.input_ids_key_name], skip_special_tokens=True)        
            all_input_texts.extend(input_texts)        
            targets = batch[self.output_ids_key_name]
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            all_outputs.extend(preds)
        else:
            # TODO: need refactor
            if self.training_kind in ["sft", "dpo"]:
                starting_indices = (batch[self.output_ids_key_name] != -100).int().argmax(dim=1)
            elif self.training_kind in ["pre"]:
                # then starting_indices is the halfway point of the input ids
                starting_indices = (batch[self.output_ids_key_name].shape[1]//2 * torch.ones(batch[self.output_ids_key_name].shape[0], dtype=torch.int)).to(batch[self.output_ids_key_name].device)
            real_input_texts = []
            real_targets = []
            real_rejecteds = []
            for j, start_idx in enumerate(starting_indices):
                if self.training_kind in ["sft", "pre"]:
                    input_ids = batch[self.input_ids_key_name][j][:start_idx]
                elif self.training_kind in ["dpo"]:
                    input_ids = batch[self.input_ids_key_name][j] # start_idx is always 0
                text = tokenizer.decode(input_ids, skip_special_tokens=True)
                real_input_texts.append(text)
                if self.training_kind in ["sft", "dpo"]:
                    output_ids = batch[self.output_ids_key_name][j][start_idx:]
                elif self.training_kind in ["pre"]:
                    output_ids = batch[self.output_ids_key_name][j][start_idx:start_idx+self.eval_max_new_tokens]
                output_text = tokenizer.decode(output_ids[output_ids != -100], skip_special_tokens=True)
                real_targets.append(output_text)
                if self.training_kind in ["dpo"]:
                    rejected_ids = batch[self.rejected_ids_key_name][j][start_idx:]
                    rejected_text = tokenizer.decode(rejected_ids[rejected_ids != -100], skip_special_tokens=True)
                    real_rejecteds.append(rejected_text)
                else:
                    real_rejecteds.append("")
            all_input_texts.extend(real_input_texts)
            all_targets.extend(real_targets)
            all_rejecteds.extend(real_rejecteds)
            current_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            inputs = tokenizer(all_input_texts, return_tensors="pt", padding=True).to(model.device)
            tokenizer.padding_side = current_padding_side
            input_length = inputs['input_ids'].shape[1]
            gen_kwargs = {"max_new_tokens": self.eval_max_new_tokens, "do_sample": False}
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, input_length:]
            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)                                                                                                                                        
            all_outputs.extend(output_texts)
        if self.training_kind in ["clf", "sft", "pre"]:
            for j, values in enumerate(zip(all_input_texts, all_targets, all_outputs)):
                input_text, target, output = values    
                self.table.add_data(state.global_step, j, input_text, target, output)
        else:
            for j, values in enumerate(zip(all_input_texts, all_targets, all_rejecteds, all_outputs)):
                input_text, target, rejected, output = values
                self.table.add_data(state.global_step, j, input_text, target, rejected, output)
        wandb.log({"Sample Outputs": self.table})
        return

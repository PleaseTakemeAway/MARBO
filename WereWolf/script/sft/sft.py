import os
import torch
import pandas as pd
import argparse
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--save_dir", default="Qwen2.5-14B-Instruct")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=16834)
    parser.add_argument("--warmup_steps", type=int, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--wandb_project", default="werewolf-sft")
    parser.add_argument("--wandb_run_name", default="accelerate-native-zero2")
    parser.add_argument("--report_to", default=os.environ.get("REPORT_TO", "none"))
    parser.add_argument("--data_path_en", default="data/sft_games/data/train_en.parquet")
    parser.add_argument("--role_data_path", default="data/sft_games/state_reconstruct/role_prediction_v3.parquet")
    return parser.parse_args()


def prepare_dataset(args):


    print("Loading datasets...")


    splits = {'train_zh': 'data/train_zh.parquet', 
              'train_en': 'data/train_en.parquet', 
              'action_zh': 'data/action_zh.parquet',
              "action": "game_behavior/action.parquet", 
              'speech_zh': 'data/speech_zh.parquet', 
              "speech": "game_behavior/speech.parquet",
              'vote_zh': 'data/vote_zh.parquet', 
              "vote": "game_behavior/vote.parquet",
              'game_strategy_and_term': 'data/game_strategy_and_term.parquet',
              'state_reconstruction': 'game_behavior/mix_data.parquet'}
    

    base_url = "data_sample/full_datasets/"

    # Dataset generation
    df_werewolf = pd.read_parquet(base_url + splits["action"])
    df_werewolf = pd.concat([df_werewolf, pd.read_parquet(base_url + splits["speech"])])
    df_werewolf = pd.concat([df_werewolf, pd.read_parquet(base_url + splits["vote"])])
    df_werewolf = pd.concat([df_werewolf, pd.read_parquet(base_url + splits["game_strategy_and_term"])])
    
    def format_werewolf(example):
        instruction = example.get("instruction")
        prompt_text = example.get("prompt")
        response_text = example.get("response")
        
        # Handle NoneType safely
        instruction = str(instruction or "").strip()
        prompt_text = str(prompt_text or "").strip()
        response_text = str(response_text or "").strip()
        
        # Concatenate instruction and prompt
        if instruction and prompt_text:
            combined_prompt = f"{instruction}\n{prompt_text}"
        else:
            combined_prompt = instruction or prompt_text
            
        return {
            "prompt": combined_prompt.strip(),
            "completion": response_text.strip()
        }

    # ── 2. SlimOrca ────────────────────────────────────────────────────────────
    try:
        slim_orca = (load_dataset("Open-Orca/SlimOrca", split="train")
                     .shuffle(seed=42).select(range(6000)))

        def format_slim_orca(example):
            # conversations: [{"from": "system"/"human"/"gpt", "value": ...}, ...]
            convs = {c["from"]: c["value"] for c in example.get("conversations", [])}
            prompt = convs.get("human", "")
            completion = convs.get("gpt", "")
            return {"prompt": prompt, "completion": completion}

        slim_orca = slim_orca.map(format_slim_orca, remove_columns=slim_orca.column_names)
        print(f"SlimOrca: {len(slim_orca)} samples")
    except Exception as e:
        print(f"SlimOrca loading failed: {e}")
        slim_orca = Dataset.from_dict({"prompt": [], "completion": []})

    # ── 3. ShareGPT ────────────────────────────────────────────────────────────
    try:
        share_gpt = (load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
                     .shuffle(seed=42).select(range(3600)))

        def format_share_gpt(example):
            convs = {c["from"]: c["value"] for c in example.get("conversations", [])}
            prompt = convs.get("human", "")
            completion = convs.get("gpt", "")
            return {"prompt": prompt, "completion": completion}

        share_gpt = share_gpt.map(format_share_gpt, remove_columns=share_gpt.column_names)
        print(f"ShareGPT: {len(share_gpt)} samples")
    except Exception as e:
        print(f"ShareGPT loading failed: {e}")
        share_gpt = Dataset.from_dict({"prompt": [], "completion": []})

    # ── 4. Alpaca ──────────────────────────────────────────────────────────────
    try:
        alpaca = (load_dataset("tatsu-lab/alpaca", split="train")
                  .shuffle(seed=42).select(range(2400)))

        def format_alpaca(example):
            input_text = f"\nInput: {example['input']}" if example.get("input") else ""
            return {"prompt": example["instruction"] + input_text, "completion": example["output"]}

        alpaca = alpaca.map(format_alpaca, remove_columns=alpaca.column_names)
        print(f"Alpaca: {len(alpaca)} samples")
    except Exception as e:
        print(f"Alpaca loading failed: {e}")
        alpaca = Dataset.from_dict({"prompt": [], "completion": []})


    if not df_werewolf.empty:
        werewolf_ds = Dataset.from_pandas(df_werewolf)
        werewolf_ds = werewolf_ds.map(format_werewolf, remove_columns=werewolf_ds.column_names)
    else:
        werewolf_ds = Dataset.from_dict({"prompt": [], "completion": []})

    combined = concatenate_datasets([werewolf_ds, slim_orca, share_gpt, alpaca])
    combined = combined.shuffle(seed=42)
    # Safely filter samples where prompt and completion are non-empty strings
    combined = combined.filter(lambda x: bool(x.get("prompt")) and bool(x.get("completion")))
    combined = combined.filter(lambda x: bool(str(x["prompt"]).strip()) and bool(str(x["completion"]).strip()))
    print(f"Combined after filtering: {len(combined)} samples")


    def to_messages(example):
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]
        }

    final = combined.map(to_messages, remove_columns=combined.column_names)
    final = final.filter(
        lambda x: x["messages"]
        and all(msg.get("content", "").strip() for msg in x["messages"])
    )
    print(f"Final training samples: {len(final)}")
    return final


# ── SFT Trainer ──────────────────────────────────────────────────
class DebugSFTTrainer(SFTTrainer):


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss = result[0] if return_outputs else result

        if torch.isnan(loss):
            print("\n[NaN Loss Detected] Printing batch samples:")
            tokenizer = self.processing_class
            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                for i, ids in enumerate(input_ids):
                    text = tokenizer.decode(ids, skip_special_tokens=False)
                    print(f"  -- sample {i} --\n{text[:500]}\n")

        return result


def main():
    args = parse_args()

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = prepare_dataset(args)

    sft_config = SFTConfig(
        output_dir=args.save_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        report_to=args.report_to,
    )

    trainer = DebugSFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Dataset size passed to trainer: {len(trainer.train_dataset)}")
    print(f"Sample[0] keys: {list(trainer.train_dataset[0].keys())}")

    trainer.train()

    print("Saving final model...")
    final_dir = os.path.join(args.save_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
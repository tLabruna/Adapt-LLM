import os
import argparse  # Import the argparse library to handle command-line arguments

def run_finetune(args):
    # Construct the command for fine-tuning, including the token as an argument
    command = (f"python3 finetune.py "
               f"--base_model {args['base_model']} "
               f"--data_path {args['train_path']} "
               f"--output_dir {args['out_dir']} "
               f"--batch_size 128 "
               f"--micro_batch_size 32 "
               f"--num_epochs 3 "
               f"--learning_rate 3e-4 "
               f"--cutoff_len 512 "
               f"--val_set_size 2000 "
               f"--lora_r 8 "
               f"--lora_alpha 16 "
               f"--lora_dropout 0.05 "
               f"--lora_target_modules '[q_proj,v_proj]' "
               f"--group_by_length ")               
    os.system(command)

if __name__ == "__main__":
    # Add command-line argument handling
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["always_retrieve", "never_retrieve", "hybrid_retrieve", "merged"], required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model", required=True)
    args = parser.parse_args()

    run_finetune(vars(args))  # Convert arguments to a dictionary and call the train function

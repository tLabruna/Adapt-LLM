import os
import argparse 

def run_evaluation(args):
    command = (f"python3 evaluate_script.py "
               f"--dataset {args['dataset']} "
               f"--mode {args['mode']} "
               f"--base_model {args['base_model']} "
               f"--data_path {args['data_path']} "
               f"--out_dir {args['out_dir']} "
               f"--lora {args['lora']} "
               f"--passages_from {args['passages_from']} "
               f"--lora_always {args['lora_always']} "
               f"--lora_never {args['lora_never']} "
               f"--hybrid_flex {args['hybrid_flex']} ")
    if args['threshold'] is not None:
        command += (f" --threshold {args['threshold']}")
    if args['from_score'] is not None:
        command += (f" --from_score {args['from_score']}")
    os.system(command)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nq", "popqa", "popqa_test", "squad", "web", "wikipedia"], required=True)
    parser.add_argument("--mode", choices=["always_retrieve", "never_retrieve", "hybrid_retrieve"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--passages_from", choices=["gold_passage", "contriever_passage", "passage_text"], required=True)
    parser.add_argument("--lora_always", required=False)
    parser.add_argument("--lora_never", required=False) 
    parser.add_argument("--hybrid_flex", required=False)
    parser.add_argument("--threshold", required=False)
    parser.add_argument("--from_score", required=False)
    args = parser.parse_args()

    run_evaluation(vars(args))
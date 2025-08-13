import argparse
import os
from metrics.metrics import eval_retrieval_qa, eval_compilation_qa, eval_definition_qa, eval_presence_qa, eval_dimensions_qa, eval_functional_performance_qa
try:
    from eval.model_router import (
        parse_model_map, choose_backend_for_subset, openai_chat, claude_chat, mock_chat
    )
except ModuleNotFoundError:
    from model_router import (
        parse_model_map, choose_backend_for_subset, openai_chat, claude_chat, mock_chat
    )

SUBSETS = ["retrieval","compilation","definition","presence","dimension","functional_performance"]

def model_map_from_env() -> str:
    """
    Build a model-map string from env vars.
    - DQ_PROVIDER sets the default (openai|anthropic|mock)
    - DQ_MODEL_<SUBSET> overrides per subset (same 3 values)
    """
    default = (os.getenv("DQ_PROVIDER") or "openai").strip().lower()
    entries = [f"default={default}"]
    for s in SUBSETS:
        env_key = f"DQ_MODEL_{s.upper()}"
        val = os.getenv(env_key)
        if val:
            entries.append(f"{s}={val.strip().lower()}")
    return ";".join(entries)

def _find_latest_csv(outputs_dir: str, keyword: str) -> str:
    """Find the most recent CSV in outputs_dir whose filename contains keyword (case-insensitive).
    Returns full path or raises FileNotFoundError.
    """
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    kw = keyword.lower()
    candidates = []
    for name in os.listdir(outputs_dir):
        if not name.lower().endswith(".csv"):
            continue
        if kw in name.lower():
            full = os.path.join(outputs_dir, name)
            try:
                mtime = os.path.getmtime(full)
            except Exception:
                mtime = 0
            candidates.append((mtime, full))
    if not candidates:
        raise FileNotFoundError(f"Could not auto-locate CSV for subset '{keyword}' in {outputs_dir}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def _auto_locate_paths(args):
    """Fill missing subset paths by auto-detecting latest CSVs in your_outputs."""
    outputs_dir = "your_outputs"
    # Map arg name to keyword to search
    pairs = [
        ("path_to_retrieval", "retrieval"),
        ("path_to_compilation", "compilation"),
        ("path_to_definition", "definition"),
        ("path_to_presence", "presence"),
        ("path_to_dimension", "dimension"),
        ("path_to_functional_performance", "functional_performance"),
    ]
    for attr, key in pairs:
        if getattr(args, attr) in (None, ""):
            try:
                auto_path = _find_latest_csv(outputs_dir, key)
                print(f"[INFO] Auto-selected {key}: {auto_path}")
                setattr(args, attr, auto_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(str(e))

def run_llm(messages, subset_name: str, model_map):
    backend = choose_backend_for_subset(subset_name, model_map)
    print(f"[INFO] Using backend='{backend}' for subset='{subset_name}'")
    if backend == "openai":
        return openai_chat(messages, temperature=0.0, max_tokens=1024)
    if backend in ("anthropic","claude"):
        return claude_chat(messages, temperature=0.0, max_tokens=1024)
    if backend == "mock":
        return mock_chat(messages, temperature=0.0, max_tokens=1024)
    raise ValueError(f"Unknown backend '{backend}' for subset '{subset_name}'")

def main():
    parser = argparse.ArgumentParser(description="Optional paths for CAD evaluation inputs")

    parser.add_argument("--path_to_retrieval", type=str, default=None,
                        help="Path to csv containing retrieval data (optional)")
    parser.add_argument("--path_to_compilation", type=str, default=None,
                        help="Path to csv containing compilation data (optional)")
    parser.add_argument("--path_to_dimension", type=str, default=None,
                        help="Path to csv containing dimension data (optional)")
    parser.add_argument("--path_to_functional_performance", type=str, default=None,
                        help="Path to csv containing functional performance data (optional)")
    parser.add_argument("--path_to_definition", type=str, default=None,
                        help="Path to csv containing definition data (optional)")
    parser.add_argument("--path_to_presence", type=str, default=None,
                        help="Path to csv containing presence data (optional)")
    parser.add_argument("--save_path", type=str, default="results.txt",
                        help="Path to .txt file to save the evaluation results (default: results.txt)")
    parser.add_argument(
        "--model-map",
        type=str,
        default=None,
        help=("Per-subset backend, e.g. "
              "'default=openai;dimension=anthropic;functional_performance=anthropic'. "
              "Valid backends: 'openai', 'anthropic', 'mock'")
    )

    args = parser.parse_args()
    model_map_str = args.model_map if args.model_map else model_map_from_env()
    model_map = parse_model_map(model_map_str)

    # Auto-detect missing CSVs from your_outputs
    _auto_locate_paths(args)

    # Check if save path already exists and ask for user confirmation
    if os.path.exists(args.save_path):
        response = input(f"File '{args.save_path}' already exists. Do you want to overwrite it? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Operation cancelled. Exiting without overwriting existing file.")
            return

    print("Arguments received:")
    print(f"  path_to_retrieval: {args.path_to_retrieval}")
    print(f"  path_to_compilation: {args.path_to_compilation}")
    print(f"  path_to_dimension: {args.path_to_dimension}")
    print(f"  path_to_functional_performance: {args.path_to_functional_performance}")
    print(f"  path_to_definition: {args.path_to_definition}")
    print(f"  path_to_presence: {args.path_to_presence}")
    
    all_subsets = []
    if args.path_to_retrieval:
        macro_avg_retrieval, all_answers_retrieval = eval_retrieval_qa(args.path_to_retrieval)
        all_subsets.append(macro_avg_retrieval)
        
    if args.path_to_compilation:
        macro_avg_compilation, all_answers_compilation = eval_compilation_qa(args.path_to_compilation)
        all_subsets.append(macro_avg_compilation)
        
    if args.path_to_definition:
        macro_avg_definition, definitions_qs_definition_avg, multi_qs_definition_avg, single_qs_definition_avg, all_answers_definition = eval_definition_qa(args.path_to_definition)
        all_subsets.append(macro_avg_definition)
        
    if args.path_to_presence:
        macro_avg_presence, definitions_qs_presence_avg, multi_qs_presence_avg, single_qs_presence_avg, all_answers_presence = eval_presence_qa(args.path_to_presence)
        all_subsets.append(macro_avg_presence)
        
    if args.path_to_dimension:
        macro_avg_accuracy_dimension, direct_dim_avg, scale_bar_avg, all_accuracies_dimension, macro_avg_bleus_dimension, all_bleus_dimension, \
                macro_avg_rogues_dimension, all_rogues_dimension = eval_dimensions_qa(args.path_to_dimension)
        all_subsets.append(macro_avg_accuracy_dimension)

    if args.path_to_functional_performance:
        macro_avg_accuracy_functional, all_accuracies_functional, macro_avg_bleus_functional, all_bleus_functional, macro_avg_rogues_functional, all_rogues_functional = eval_functional_performance_qa(args.path_to_functional_performance)
        all_subsets.append(macro_avg_accuracy_functional)

    # Write all the results to a file
    with open(args.save_path, 'w') as text_file:
        text_file.write("DESIGNQA EVALUATION RESULTS:\n")
        text_file.write("-* -" * 20 + "\n")
        text_file.write("-* -" * 20 + "\n")
        text_file.write(f"OVERALL SCORE: {sum(all_subsets) / 6}\n")
        text_file.write("-* -" * 20 + "\n")
        text_file.write("-* -" * 20 + "\n")
        text_file.write(f"Retrieval Score (Avg F1 BoW): {macro_avg_retrieval if args.path_to_retrieval else 'N/A'}\n")
        text_file.write(f"Compilation Score (Avg F1 Rules): {macro_avg_compilation if args.path_to_compilation else 'N/A'}\n")
        text_file.write(f"Definition Score (Avg F1 BoC): {macro_avg_definition if args.path_to_definition else 'N/A'}\n")
        text_file.write(f"Presence Score (Avg Accuracy): {macro_avg_presence if args.path_to_presence else 'N/A'}\n")
        text_file.write(f"Dimension Score (Average Accuracy): {macro_avg_accuracy_dimension if args.path_to_dimension else 'N/A'}\n")
        text_file.write(f"Functional Performance Score (Average Accuracy): {macro_avg_accuracy_functional if args.path_to_functional_performance else 'N/A'}\n")
        text_file.write("-* -" * 20 + "\n")
        text_file.write("\n\n\n")
        text_file.write("Below scores by subset are provided for diagnostic purposes:\n")
        text_file.write("---" * 20 + "\n")
        text_file.write("RETRIEVAL\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_retrieval:
            text_file.write(f"All F1 BoWs:\n{all_answers_retrieval}\n")
        else:
            text_file.write("No retrieval data provided.\n")
        
        text_file.write("---" * 20 + "\n")
        text_file.write("COMPILATION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_compilation:
            text_file.write(f"All F1 Rules:\n{all_answers_compilation}\n")
        else:
            text_file.write("No compilation data provided.\n")
        
        text_file.write("---" * 20 + "\n")
        text_file.write("DEFINITION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_definition:
            text_file.write(f"Avg F1 BoC on definition-components:\n{definitions_qs_definition_avg}\n")
            text_file.write(f"Avg F1 BoC on multimention-components:\n{multi_qs_definition_avg}\n")
            text_file.write(f"Avg F1 BoC on no-mention-components:\n{single_qs_definition_avg}\n")
            text_file.write(f"All F1 BoC:\n{all_answers_definition}\n")
        else:
            text_file.write("No definition data provided.\n")
            
        text_file.write("---" * 20 + "\n")
        text_file.write("PRESENCE\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_presence:
            text_file.write(f"Avg accuracy on definition-components:\n{definitions_qs_presence_avg}\n")
            text_file.write(f"Avg accuracy on multimention-components:\n{multi_qs_presence_avg}\n")
            text_file.write(f"Avg accuracy on no-mention-components:\n{single_qs_presence_avg}\n")
            text_file.write(f"All accuracies:\n{all_answers_presence}\n")
        else:
            text_file.write("No presence data provided.\n")

        text_file.write("---" * 20 + "\n")
        text_file.write("DIMENSION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_dimension:
            text_file.write(f"Avg accuracy directly-dimensioned:\n{direct_dim_avg}\n")
            text_file.write(f"Avg accuracy scale-bar-dimensioned:\n{scale_bar_avg}\n")
            text_file.write(f"All accuracies:\n{all_accuracies_dimension}\n")
            text_file.write(f"Avg BLEU score:\n{macro_avg_bleus_dimension}\n")
            text_file.write(f"All BLEU scores:\n{all_bleus_dimension}\n")
            text_file.write(f"Avg ROUGE score:\n{macro_avg_rogues_dimension}\n")
            text_file.write(f"All ROUGE scores:\n{all_rogues_dimension}\n")
        else:
            text_file.write("No dimension data provided.\n")
            
        text_file.write("---" * 20 + "\n")
        text_file.write("FUNCTIONAL PERFORMANCE\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_functional_performance:
            text_file.write(f"All accuraciess:\n{all_accuracies_functional}\n")
            text_file.write(f"Avg BLEU score:\n{macro_avg_bleus_functional}\n")
            text_file.write(f"All BLEU scores:\n{all_bleus_functional}\n")
            text_file.write(f"Avg ROUGE score:\n{macro_avg_rogues_functional}\n")
        else:
            text_file.write("No functional performance data provided.\n")

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    







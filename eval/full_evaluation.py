import argparse
import os
import pandas as pd
from eval.metrics.metrics import eval_retrieval_qa, eval_compilation_qa, eval_definition_qa, eval_presence_qa, eval_dimensions_qa, eval_functional_performance_qa

def validate_csv_has_model_prediction(csv_path, subset_name):
    """
    Validate that a CSV file contains the required 'model_prediction' column.
    
    Args:
        csv_path (str): Path to the CSV file
        subset_name (str): Name of the subset for error reporting
        
    Returns:
        bool: True if validation passes, False if model_prediction column is missing
    """
    try:
        df = pd.read_csv(csv_path)
        if 'model_prediction' not in df.columns:
            print(f"[WARN] Skipping {subset_name}: {csv_path} has no 'model_prediction' column (likely a dataset-only CSV).")
            return False
        print(f"✓ {subset_name} CSV validation passed: 'model_prediction' column found")
        return True
    except FileNotFoundError:
        print(f"[ERROR] Could not find the CSV file: {csv_path}")
        return False
    except Exception as e:
        print(f"[ERROR] Error reading CSV file {csv_path}: {str(e)}")
        return False

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

    args = parser.parse_args()

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
    
    # Initialize variables to avoid NameError if validation fails
    macro_avg_retrieval = all_answers_retrieval = None
    macro_avg_compilation = all_answers_compilation = None
    macro_avg_definition = definitions_qs_definition_avg = multi_qs_definition_avg = single_qs_definition_avg = all_answers_definition = None
    macro_avg_presence = definitions_qs_presence_avg = multi_qs_presence_avg = single_qs_presence_avg = all_answers_presence = None
    macro_avg_accuracy_dimension = direct_dim_avg = scale_bar_avg = all_accuracies_dimension = macro_avg_bleus_dimension = all_bleus_dimension = macro_avg_rogues_dimension = all_rogues_dimension = None
    macro_avg_accuracy_functional = all_accuracies_functional = macro_avg_bleus_functional = all_bleus_functional = macro_avg_rogues_functional = all_rogues_functional = None
    
    if args.path_to_retrieval:
        if validate_csv_has_model_prediction(args.path_to_retrieval, "retrieval"):
            macro_avg_retrieval, all_answers_retrieval = eval_retrieval_qa(args.path_to_retrieval)
            all_subsets.append(macro_avg_retrieval)
        
    if args.path_to_compilation:
        if validate_csv_has_model_prediction(args.path_to_compilation, "compilation"):
            macro_avg_compilation, all_answers_compilation = eval_compilation_qa(args.path_to_compilation)
            all_subsets.append(macro_avg_compilation)
        
    if args.path_to_definition:
        if validate_csv_has_model_prediction(args.path_to_definition, "definition"):
            macro_avg_definition, definitions_qs_definition_avg, multi_qs_definition_avg, single_qs_definition_avg, all_answers_definition = eval_definition_qa(args.path_to_definition)
            all_subsets.append(macro_avg_definition)
        
    if args.path_to_presence:
        if validate_csv_has_model_prediction(args.path_to_presence, "presence"):
            macro_avg_presence, definitions_qs_presence_avg, multi_qs_presence_avg, single_qs_presence_avg, all_answers_presence = eval_presence_qa(args.path_to_presence)
            all_subsets.append(macro_avg_presence)
        
    if args.path_to_dimension:
        if validate_csv_has_model_prediction(args.path_to_dimension, "dimension"):
            macro_avg_accuracy_dimension, direct_dim_avg, scale_bar_avg, all_accuracies_dimension, macro_avg_bleus_dimension, all_bleus_dimension, \
                    macro_avg_rogues_dimension, all_rogues_dimension = eval_dimensions_qa(args.path_to_dimension)
            all_subsets.append(macro_avg_accuracy_dimension)

    if args.path_to_functional_performance:
        if validate_csv_has_model_prediction(args.path_to_functional_performance, "functional_performance"):
            macro_avg_accuracy_functional, all_accuracies_functional, macro_avg_bleus_functional, all_bleus_functional, macro_avg_rogues_functional, all_rogues_functional = eval_functional_performance_qa(args.path_to_functional_performance)
            all_subsets.append(macro_avg_accuracy_functional)

    # Write all the results to a file
    with open(args.save_path, 'w') as text_file:
        text_file.write("DESIGNQA EVALUATION RESULTS:\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write(f"OVERALL SCORE: {sum(all_subsets) / 6}\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write(f"Retrieval Score (Avg F1 BoW): {macro_avg_retrieval if macro_avg_retrieval is not None else 'N/A'}\n")
        text_file.write(f"Compilation Score (Avg F1 Rules): {macro_avg_compilation if macro_avg_compilation is not None else 'N/A'}\n")
        text_file.write(f"Definition Score (Avg F1 BoC): {macro_avg_definition if macro_avg_definition is not None else 'N/A'}\n")
        text_file.write(f"Presence Score (Avg Accuracy): {macro_avg_presence if macro_avg_presence is not None else 'N/A'}\n")
        text_file.write(f"Dimension Score (Average Accuracy): {macro_avg_accuracy_dimension if macro_avg_accuracy_dimension is not None else 'N/A'}\n")
        text_file.write(f"Functional Performance Score (Average Accuracy): {macro_avg_accuracy_functional if macro_avg_accuracy_functional is not None else 'N/A'}\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("\n\n\n")
        text_file.write("Below scores by subset are provided for diagnostic purposes:\n")
        text_file.write("---" * 20 + "\n")
        text_file.write("RETRIEVAL\n")
        text_file.write("---" * 20 + "\n")
        if macro_avg_retrieval is not None:
            text_file.write(f"All F1 BoWs:\n{all_answers_retrieval}\n")
        else:
            text_file.write("No retrieval data provided.\n")
        
        text_file.write("---" * 20 + "\n")
        text_file.write("COMPILATION\n")
        text_file.write("---" * 20 + "\n")
        if macro_avg_compilation is not None:
            text_file.write(f"All F1 Rules:\n{all_answers_compilation}\n")
        else:
            text_file.write("No compilation data provided.\n")
        
        text_file.write("---" * 20 + "\n")
        text_file.write("DEFINITION\n")
        text_file.write("---" * 20 + "\n")
        if macro_avg_definition is not None:
            text_file.write(f"Avg F1 BoC on definition-components:\n{definitions_qs_definition_avg}\n")
            text_file.write(f"Avg F1 BoC on multimention-components:\n{multi_qs_definition_avg}\n")
            text_file.write(f"Avg F1 BoC on no-mention-components:\n{single_qs_definition_avg}\n")
            text_file.write(f"All F1 BoC:\n{all_answers_definition}\n")
        else:
            text_file.write("No definition data provided.\n")
            
        text_file.write("---" * 20 + "\n")
        text_file.write("PRESENCE\n")
        text_file.write("---" * 20 + "\n")
        if macro_avg_presence is not None:
            text_file.write(f"Avg accuracy on definition-components:\n{definitions_qs_presence_avg}\n")
            text_file.write(f"Avg accuracy on multimention-components:\n{multi_qs_presence_avg}\n")
            text_file.write(f"Avg accuracy on no-mention-components:\n{single_qs_presence_avg}\n")
            text_file.write(f"All accuracies:\n{all_answers_presence}\n")
        else:
            text_file.write("No presence data provided.\n")

        text_file.write("---" * 20 + "\n")
        text_file.write("DIMENSION\n")
        text_file.write("---" * 20 + "\n")
        if macro_avg_accuracy_dimension is not None:
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
        if macro_avg_accuracy_functional is not None:
            text_file.write(f"All accuraciess:\n{all_accuracies_functional}\n")
            text_file.write(f"Avg BLEU score:\n{macro_avg_bleus_functional}\n")
            text_file.write(f"All BLEU scores:\n{all_bleus_functional}\n")
            text_file.write(f"Avg ROUGE score:\n{macro_avg_rogues_functional}\n")
            text_file.write(f"All ROUGE scores:\n{all_rogues_functional}\n")
        else:
            text_file.write("No functional performance data provided.\n")

if __name__ == "__main__":
    main()
    







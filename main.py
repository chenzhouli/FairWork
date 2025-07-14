# main.py
import os
import json
import argparse
from typing import Dict

from functions.llm_services import LLMProvider, load_llm_config
from group_function import prepare_analysis_data, run_full_analysis
from config import DEFAULT_SENSITIVE_VALUES, DATA_DIR


class DummyEvent:
    def is_set(self):
        return False

def save_results_to_json(results: Dict, output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Analysis successful. Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Group Fairness Analysis without UI.")
    parser.add_argument('--models', nargs='+', required=True, help="A list of model names to evaluate, as defined in config.yaml. e.g., --models 'Llama3 (via Groq)' 'GPT-4o'")
    parser.add_argument('--users_file', type=str, default=str(DATA_DIR / "users_150.tsv"), help="Path to the users dataset.")
    parser.add_argument('--jobs_file', type=str, default=str(DATA_DIR / "jobs_150.tsv"), help="Path to the jobs dataset.")
    parser.add_argument('--apps_file', type=str, default=str(DATA_DIR / "apps.tsv"), help="Path to the applications dataset.")
    parser.add_argument('--num_users', type=int, default=50, help="Number of users to sample for the analysis.")
    parser.add_argument('--jobs_per_user', type=int, default=10, help="Number of jobs to sample per user.")
    parser.add_argument('--output_file', type=str, default="results/group_analysis_results.json", help="Path to save the output JSON file.")
    
    args = parser.parse_args()

    print("Initializing LLM Provider and loading configurations...")
    llm_provider = LLMProvider()
    available_llms = load_llm_config()
    
    selected_configs = [llm for llm in available_llms if llm['name'] in args.models]
    if not selected_configs:
        print(f"Error: None of the specified models {args.models} were found in config.yaml.")
        return

    print(f"Found {len(selected_configs)} models to evaluate: {[config['name'] for config in selected_configs]}")

    print("\n--- Stage 1: Preparing analysis data (this happens only once) ---")
    try:
        sensitive_attributes = {
            attr: [v.strip() for v in values.split(',')]
            for attr, values in DEFAULT_SENSITIVE_VALUES.items()
        }
        
        prepared_data = prepare_analysis_data(
            users_file=args.users_file,
            jobs_file=args.jobs_file,
            apps_file=args.apps_file,
            num_users=args.num_users,
            N=args.jobs_per_user,
            pos_ratio=0.4,
            cancel_event=DummyEvent()
        )
        print("Data preparation complete.")
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return

    print("\n--- Stage 2: Running analysis for each selected model ---")
    all_models_results = {}
    for config in selected_configs:
        model_name = config['name']
        print(f"\nProcessing model: {model_name}")

        def single_model_llm_call(prompt: str):
            return llm_provider.get_llm_response(config, prompt)
        
        try:
            result_for_one_model = run_full_analysis(
                prepared_data=prepared_data,
                llm_function=single_model_llm_call,
                threshold=5.0,
                sensitive_attr_dict=sensitive_attributes,
                cancel_event=DummyEvent()
            )
            all_models_results[model_name] = result_for_one_model
            print(f"Finished analysis for {model_name}.")
        except Exception as e:
            print(f"An error occurred while processing model {model_name}: {e}")
            all_models_results[model_name] = {"error": str(e)}

    print("\n--- Stage 3: Aggregating and saving final results ---")
    final_output = {
        "analysis_parameters": {
            "models_evaluated": args.models,
            "num_users": args.num_users,
            "jobs_per_user": args.jobs_per_user,
        },
        "attr_dict": sensitive_attributes,
        "original_predictions": prepared_data.get('original_predictions'),
        "analysis_results": all_models_results,
        "overview_metrics": {name: data.get("overview_metrics", {}) for name, data in all_models_results.items() if data},
        "overview_bias": {name: data.get("overview_bias", {}) for name, data in all_models_results.items() if data}
    }

    save_results_to_json(final_output, args.output_file)

if __name__ == "__main__":
    main()
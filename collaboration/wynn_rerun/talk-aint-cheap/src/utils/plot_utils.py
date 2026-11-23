import json
import yaml
from datetime import datetime
import os


def get_single_and_multi_agent_results(base_output_dir, dataset):
    single_agent_path = get_most_recent_experiment_path(base_output_dir, dataset, {'num_agents':1, 'num_rounds':0})
    multi_agent_path = get_most_recent_experiment_path(base_output_dir, dataset, {'num_agents':3, 'num_rounds':3})

    single_agent_results, multi_agent_results = None, None
    if single_agent_path is not None:
        with open(single_agent_path + dataset + '_performance.json', "r") as f:
            single_agent_results = json.load(f)

    if multi_agent_path is not None:
        with open(multi_agent_path + dataset + '_performance.json', "r") as f:
            multi_agent_results = json.load(f)

    return single_agent_results, multi_agent_results


def get_final_acc(results):
    return next(iter(results[-1].values())) * 100


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None
            

# Conditions: For now, only num_agents, num_rounds, num_questions, summarize are supported
def get_most_recent_experiment_path(output_path, dataset, conditions=None):
    assert dataset in ['mmlu', 'math', 'commonsense_qa']
    
    # Get a list of all the output folders as a list of datetime strings
    datetimes = []
    date_strings = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
    for date in date_strings:
        date_path = output_path + date + '/'
        time_strings = [f for f in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, f))]
        for time in time_strings:
            datetimes.append(date + ' ' + time)
            
    # Sort the datetime strings such that the most recent is first
    dates = [datetime.strptime(date_str, '%Y-%m-%d %H-%M-%S') for date_str in datetimes]
    dates.sort(reverse=True)
    
    # Look for the first folder with data for this dataset
    for date in dates:
        # Get folder name based on datetime
        date_folder = date.strftime('%Y-%m-%d %H-%M-%S').replace(' ', '/') + '/'
        path = output_path + date_folder
        if os.path.exists(path + dataset + '_performance.json'):
            if conditions is not None:
                # Check that the experiment conditions are also satisfied
                config = read_yaml_file(path + '.hydra/config.yaml')
                conditions_match = True
                for condition in conditions:
                    if config['experiment'][condition] != conditions[condition]:
                        conditions_match = False
                # if all conditions are satisfied, return the path
                if conditions_match:
                    return path
            else:
                # Just immediately return the folder if there's no conditions to check
                return path

    print('No results for dataset', dataset)
    if conditions is not None:
        print('Conditions:', conditions)
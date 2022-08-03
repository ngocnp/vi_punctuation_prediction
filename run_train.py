import json
from model_trainer import ModelTrainer
import sys

from utils import load_json


if __name__ == "__main__":

    # read model configs
    suite_config = load_json(f'model_final_suite.json')
    completed_runs = load_json(f'model_final_suite_results.json')

    # invoke training
    for run_config in suite_config["tests"]:

        run_name = f'{run_config["model"].replace("/", "-")}-{"-".join(run_config["languages"])}-{run_config["data_percentage"]}-task{run_config["task"]}'
        if "comment" in run_config:
            run_name += "-" + run_config["comment"]

        if str(run_config["id"]) not in completed_runs["tests"]:
            print("invoking test run for model: " + run_name)
        else:
            print("skipping finished run for model: " + run_name)
            continue

        try:
            trainer = ModelTrainer(run_name=run_name, **run_config)
            result = trainer.run_training()
            run_config["result"] = result
            completed_runs["tests"][run_config["id"]] = run_config
        except:
            print("Unexpected error:", sys.exc_info()[0])

            # write results
        with open(f'model_final_suite_results.json', 'w') as outfile:
            json.dump(completed_runs, outfile, indent=4)


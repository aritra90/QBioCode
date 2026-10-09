# ====== Base class imports ======
import os
import json
import pandas as pd
import subprocess
import yaml
import glob
import argparse
from datetime import datetime, timezone
import time

# ======= Parallelization =======
from joblib import Parallel, delayed

# ======= checkpointing =========
from qbiocode import checkpoint_restart


def config_job_name(data_type, output_folder_timestamp, data_file):
    """The name of the per-dataset config ``run_job`` writes under ``configs/``.

    One function because two places need the same string: ``run_job``, which writes the
    file, and ``main``, which removes it afterwards. They disagreed -- the cleanup globbed
    ``configs/config_<timestamp>*`` while the files are called
    ``config_<data_type>_<timestamp>__<dataset>.yaml`` -- so the glob matched nothing and
    every batch run left one config per dataset behind for good.
    """
    stem = data_file.replace('.csv', '').replace('.txt', '')
    return f"config_{data_type}_{output_folder_timestamp}__{stem}"


def run_job(data_file, configfile, output_folder_timestamp, data_type):
    """Run QProfiler on a single dataset file with custom configuration.
    
    This function creates a temporary configuration file for each dataset by:
    1. Loading the base configuration from the specified config file
    2. Adding dataset-specific parameters (filename, timestamp, data type)
    3. Saving a new config file with a unique name
    4. Executing qprofiler with the custom configuration
    5. Cleaning up temporary config files after processing
    
    This function is designed for batch processing where multiple datasets are
    processed in parallel, each with its own configuration variant.

    Args:
        data_file (str): Name of the CSV data file to process (e.g., 'dataset1.csv')
        configfile (str): Path to the base YAML configuration file to use as template
        output_folder_timestamp (str): Timestamp string for organizing output directories
        data_type (str): Label for this batch of data (used in output directory naming)
        
    Returns:
        int: the exit status of the qprofiler process. Non-zero means that dataset
        failed; ``main`` counts those rather than reporting success for all of them.

    Example:
        >>> run_job('cancer_data.csv', 'configs/base.yaml', '2024_01_15_120000', 'cancer_study')
        # Creates configs/config_cancer_study_2024_01_15_120000__cancer_data.yaml
        # Runs: qprofiler --config-name=config_cancer_study_2024_01_15_120000__cancer_data
    """

    ## edit YAML

    # Read the YAML file. Opened 'r', not 'r+': nothing is written back through this
    # handle (the modified config goes to a new file below), and 'r+' additionally
    # refused a read-only base config -- a reasonable thing for a frozen protocol to be.
    with open(configfile, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
        # add timestamp to output dir key of config file
        data['timestamp'] = output_folder_timestamp
        data['data_type'] = data_type

    # Modify the entry
    data["file_dataset"] = data_file

    # Write the updated data back to the file
    config_name = config_job_name(data_type, output_folder_timestamp, data_file)
    config_dir = os.path.abspath('configs')
    config_file = os.path.join(config_dir, config_name + '.yaml')

    # Ensure configs directory exists
    os.makedirs(config_dir, exist_ok=True)

    with open(config_file, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    commands = ["qprofiler", f"--config-dir={config_dir}", f"--config-name={config_name}"]
    # The status is returned rather than discarded. `subprocess.run(commands)` without it
    # made a batch of failing datasets look exactly like a batch of successful ones: the
    # only hint was the "Results not found" line in the collection step below, which also
    # appears when the results are merely somewhere else.
    done = subprocess.run(commands)
    if done.returncode != 0:
        print(f"!! qprofiler exited {done.returncode} for {data_file} "
              f"(config {config_name})")
    return done.returncode


def parse_args():
    """Parse command-line arguments for batch mode processing.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='QProfiler Batch Mode - Process multiple datasets in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  qprofiler-batch
  
  # Custom input directory and config
  qprofiler-batch --input-dir data/my_datasets --config configs/my_config.yaml
  
  # Parallel processing with 4 jobs
  qprofiler-batch --input-dir data/datasets --n-jobs 4
  
  # Resume from previous run
  qprofiler-batch --input-dir data/datasets --checkpoint results/batch_2024_01_15
  
  # Custom data type label
  qprofiler-batch --input-dir data/cancer_data --data-type cancer_study
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/tutorial_test_data/lower_dim_datasets',
        help='Path to directory containing input CSV datasets (default: data/tutorial_test_data/lower_dim_datasets)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/basic_config.yaml',
        help='Path to base configuration YAML file (default: configs/basic_config.yaml)'
    )
    
    parser.add_argument(
        '--data-type',
        type=str,
        default='test_data',
        help='Label for this batch of data (used in output directory naming) (default: test_data)'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs to run (default: 1)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to previous results directory to resume from (optional)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run qprofiler in batch mode. It sets up the environment, processes datasets in parallel, and collects results.
    This function is designed to handle multiple datasets efficiently, allowing for parallel processing of machine learning methods and datasets.
    
    Args:
        None (uses command-line arguments)
    Returns:
        None
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up parameters from arguments
    input_data_path = args.input_dir
    configfile = args.config
    data_type = args.data_type
    n_jobs = args.n_jobs
    checkpoint_dir = args.checkpoint
    
    # Generate timestamp for this batch run
    output_folder_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H_%M_%S_%f")
    beg_time = time.time()
    
    # Validate inputs
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Input directory not found: {input_data_path}")
    
    if not os.path.exists(configfile):
        raise FileNotFoundError(f"Config file not found: {configfile}")
    
    current_dir = os.getcwd()
    path_to_input = os.path.join(current_dir, input_data_path)
    
    print(f"QProfiler Batch Mode")
    print(f"=" * 60)
    print(f"Input directory: {input_data_path}")
    print(f"Config file: {configfile}")
    print(f"Data type: {data_type}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Output timestamp: {output_folder_timestamp}")
    if checkpoint_dir:
        print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"=" * 60)
    
    # Handle checkpoint restart if specified
    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
            print("Proceeding without checkpoint...")
            completed_files = []
        else:
            print(f"Resuming from checkpoint: {checkpoint_dir}")
            completed_files = checkpoint_restart(checkpoint_dir, verbose=True)
            print(f"Found {len(completed_files)} completed datasets")
        
        # Process only incomplete datasets
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_job)(file, configfile, output_folder_timestamp, data_type)
            for file in os.listdir(path_to_input)
            if file.endswith('csv') and file not in completed_files
        )
    else:
        # Process all datasets
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_job)(file, configfile, output_folder_timestamp, data_type)
            for file in os.listdir(path_to_input)
            if file.endswith('csv')
        )
    
    # Collect results
    print("\nCollecting results...")
    final_model_results = pd.DataFrame()
    final_rde_results = pd.DataFrame()
    failed = [code for code in (results or []) if code]
    output_dir = f'results/{data_type}_batch_{output_folder_timestamp}'

    for file in os.listdir(path_to_input):
        if not file.endswith('csv'):
            continue
        # Searched for rather than computed. This used to read exactly
        #   results/<type>_batch_<stamp>/dataset=<file>/ModelResults.csv
        # which is only where the file lands if the config's hydra.run.dir ends at the
        # dataset level -- the packaged config.yaml adds a '<backend>_<timestamp>' level
        # below it, so with any shipped config nothing was ever found and every dataset
        # printed "Results not found" while its results sat one directory deeper.
        found = sorted(glob.glob(
            os.path.join(output_dir, f'dataset={file}', '**', 'ModelResults.csv'),
            recursive=True,
        ))
        if not found:
            print(f"Warning: Results not found for {file} under {output_dir}")
            continue
        # The newest run directory of that dataset, by the timestamp in its name.
        indv_results = found[-1]
        print(f"Processing results for: {file}  ({indv_results})")
        final_model_results = pd.concat(
            [final_model_results, pd.read_csv(indv_results, index_col=0)]
        )
        rde = os.path.join(os.path.dirname(indv_results), 'RawDataEvaluation.csv')
        if os.path.isfile(rde):
            final_rde_results = pd.concat([final_rde_results, pd.read_csv(rde, index_col=0)])

    # Clean up the per-dataset configs this batch wrote -- once, and by the name they
    # were actually written under (config_job_name).
    for file in os.listdir(path_to_input):
        if file.endswith('csv'):
            stale = os.path.join(
                'configs',
                config_job_name(data_type, output_folder_timestamp, file) + '.yaml',
            )
            if os.path.isfile(stale):
                os.remove(stale)

    # Save combined results. NOT inside the dataset directories they were read from:
    # the merged file sits at the root of the batch directory, so a later glob for
    # '**/ModelResults.csv' under a dataset does not pick it up alongside its parts.
    os.makedirs(output_dir, exist_ok=True)

    final_model_results.to_csv(f'{output_dir}/ModelResults.csv')
    final_rde_results.to_csv(f'{output_dir}/RawDataEvaluation.csv')

    total_time = (time.time() - beg_time) / 3600
    print(f"\n{'=' * 60}")
    print("Batch processing complete!" if not failed
          else f"Batch finished with {len(failed)} FAILED dataset(s) -- see the lines above")
    print(f"Total run time: {round(total_time, 2)} hours")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")

    return None


if __name__ == "__main__":
    main()


import os

"""
This code searches all the instances of batch_error.log and looks if the jobs exceeded the nodes's timelimit. This helps me from navigating 75 different folder and check which one failed due to the time limit.
The os.walk() takes a lot of time, so we save a copy of searched batch_error.log to "paths.txt" to save our time on subsequent runs.

"""

def search_file(filename):
    # Search all batch_error.log files
    results = []
    search_path = os.getcwd()  # Get the current working directory
    
    for root, _, files in os.walk(search_path):
        if filename in files:
            file_path = os.path.join(root, filename)
            results.append(file_path)
    
    return results

def save_results_to_file(results, output_file):
    # a generic function to store a list on paths in a file with \n separator
    with open(output_file, 'w') as file:
        for file_path in results:
            file.write(file_path + '\n')

def load_results_from_file(input_file):
    # Load the list of paths separated by \n and load them in a list of strings
    results = []
    with open(input_file, 'r') as file:
        for line in file:
            results.append(line.strip())
    return results

def compare_with_template(file_path, template_file):
    """
    Compares all batch_error.log files with a standard hydra error which is normal and isn't actually and error.
    If there is anything in the batch_error.log file, it will store the location of the file in the provided file_path.

    This is the standard hydra error: /scratch/s.1915438/modulus_pysdf/modulus_pysdf/lib/python3.8/site-packages/requests/__init__.py:109: RequestsDependencyWarning: urllib3 (1.26.9) or chardet (5.0.0)/charset_normalizer (2.0.12) doesn't match a supported version!
    warnings.warn(
    """
    with open(template_file, 'r') as template:
        template_error = template.read()

    with open(file_path, 'r') as file:
        content = file.read()

    return content != template_error

def search_and_save_error_files(filename, output_file, template_file):
    """
    Save all the batch_error.log paths which includes timelimit, c++ error, pytorch warning etc.
    """
    if not os.path.isfile(output_file):
        # Results file does not exist, perform the search
        results = search_file(filename)
        save_results_to_file(results, output_file)
    else:
        # Results file exists, load the results
        results = load_results_from_file(output_file)

    error_files_list = []

    for file_path in results:
        if compare_with_template(file_path, template_file):
            error_files_list.append(file_path)

    if error_files_list:
        save_results_to_file(error_files_list, output_file)
        print("Found {} instances of the file with errors. Error files saved to {}.".format(len(error_files_list), output_file))
    else:
        print("No instances of the file were found with errors.")

def search_and_save_slurm_error_files(filename, output_error_file, additional_search_keyword):
    """
    Save all the batch_error.log paths which contains timelimit error (to reduce the search space) denoted by the text "slurmstepd".
    """
    error_files = load_results_from_file(filename) # list of all error files

    slurm_error_files_list = []

    for file_path in error_files:
        if additional_search_keyword in open(file_path).read():
            slurm_error_files_list.append(file_path)

    if slurm_error_files_list:
        save_results_to_file(slurm_error_files_list, output_error_file)
        print("Found {} instances of the file with errors containing '{}'. Error files saved to {}.".format(len(slurm_error_files_list), additional_search_keyword, output_error_file))
    else:
        print("No instances of the file were found with errors containing '{}'.".format(additional_search_keyword))


def main():
    filename = "batch_error.log" # created in each directory during jobs submission
    output_file = "paths.txt" # save paths of all batch_error.log
    error_files = "error_files.txt" # save paths of all batch_error.log with warning and error
    template_file = "template_error.log" # template warning to comapre each file with
    additional_search_keyword = "slurmstepd" # slurm timelimit text to search for
    slurm_error_file = "slurm_error.txt" # write the paths of each batch_error.log with timelimit error

    search_and_save_error_files(filename, output_file, template_file)
    search_and_save_slurm_error_files(error_files, slurm_error_file, additional_search_keyword)


if __name__ == "__main__":
    main()


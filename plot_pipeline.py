import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

def main(dir):
    '''
        Main function to run the pipeline. 
        - Takes as argument a directory (dir)
        Iniside the dir should be all the config.yaml files to run.
    '''
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Call get_all_data() and take all the data required to make the plots
    (aggregated_batch_sizes, aggregated_prefill_mean_latencies, aggregated_decode_mean_latencies,
     aggregated_prefill_max_ram, aggregated_decode_max_ram, aggregated_decode_throughput,
     aggregated_prefill_throughput) = get_all_data(dir)

    # Array of all the plotting functions
    plot_functions = [
        plot_decode_memory,
        plot_prefill_memory,
        plot_decode_latency,
        plot_prefill_latency,
        plot_decode_throughput,
        plot_prefill_throughput
    ]
    plot_args = [
        (aggregated_batch_sizes, aggregated_decode_max_ram),
        (aggregated_batch_sizes, aggregated_prefill_max_ram),
        (aggregated_batch_sizes, aggregated_decode_mean_latencies),
        (aggregated_batch_sizes, aggregated_prefill_mean_latencies),
        (aggregated_batch_sizes, aggregated_decode_throughput),
        (aggregated_batch_sizes, aggregated_prefill_throughput)
    ]
    plot_titles = [
        "decode_memory",
        "prefill_memory",
        "decode_latency",
        "prefill_latency",
        "decode_throughput",
        "prefill_throughput"
    ]
    # Look through the necessary parameters to create all the plots and save them to the right dir
    for func, args, title in zip(plot_functions, plot_args, plot_titles):
        plt.figure()
        func(*args)
        plt.savefig(f"{plots_dir}/{title}.png")
        plt.close()

def load_and_process_all_files(config_path, result_path):
    '''
        Function to load the .json config and results files that are generated after each HOB experiments, and pre-process the data for the plots.
        Arguments:
        - config_path: path to the .json file containing the config of the experiment ran
        - result_path: path to the .json file containing the results of the experiment ran
    '''

    # Read the .json files and filter out the necessary information into separate dataframes
    try:    
        config_df = pd.read_json(config_path)
        result_df = pd.read_json(result_path)
    except FileNotFoundError as e:
        print("The configuration file or result file was not found {e}")
        return None

    try:
        prefill_memory_df = pd.json_normalize(result_df['prefill']['memory'])
        prefill_latency_df = pd.json_normalize(result_df['prefill']['latency'])
        prefill_throughput_df = pd.json_normalize(result_df['prefill']['throughput'])
        decode_memory_df = pd.json_normalize(result_df['decode']['memory'])
        decode_latency_df = pd.json_normalize(result_df['decode']['latency'])
        decode_throughput_df = pd.json_normalize(result_df['decode']['throughput'])
    except KeyError as e:
        print(f"Error: Missing expected data key in results {e}")
        return None
    # Pre-process te config file to remove all the NaNs. 
    try:
        input_shapes_df = pd.json_normalize(config_df['benchmark'])
        cleaned_df = input_shapes_df.dropna(how='all').ffill().bfill()
    
        batch_size = cleaned_df['batch_size'].iloc[0]
        mean_prefill_latency = prefill_latency_df['mean'].iloc[0]
        mean_decode_latency = decode_latency_df['mean'].iloc[0]
        max_ram_prefill = prefill_memory_df['max_ram'].iloc[0]
        max_ram_decode = decode_memory_df['max_ram'].iloc[0]
        decode_throughput = decode_throughput_df['value'].iloc[0]
        prefill_throughput = prefill_throughput_df['value'].iloc[0]

        return batch_size, mean_prefill_latency, mean_decode_latency, max_ram_prefill, max_ram_decode, decode_throughput, prefill_throughput
    except Exception as e:
        print(f"An error occured during the pre-processing of data {e}")
        return None


def get_all_data(dir):    
    '''
        Function to get all the data from each experiment ran that are located inside dir. 
        Arguments:
        - dir: output directory of the experiments.
    '''
    try:
        if dir is not None:
            base_dir = dir

    except ValueError as e:
        print(f"Error: No directory was given or directory {dir} does not exist {e}")
        return None

    # initialise lists for data
    aggregated_batch_sizes = []
    aggregated_prefill_mean_latencies = []
    aggregated_decode_mean_latencies = []
    aggregated_prefill_max_ram = []
    aggregated_decode_max_ram = []
    aggregated_decode_throughput = []
    aggregated_prefill_throughput = []


    # loop thorugh sub directories of dir
    for subdir_name in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir_name)
        if os.path.isdir(subdir_path):
            #path to config and results
            config_path = os.path.join(subdir_path, "experiment_config.json")
            result_path = os.path.join(subdir_path, "benchmark_report.json")
            
            if os.path.exists(config_path) and os.path.exists(result_path):

                # Call the load and process all files to clean the data and make it suitable for plotting. 
                batch_size, prefill_mean_latency, decode_mean_latency, prefill_max_ram, decode_max_ram, decode_throughput, prefill_throughput = load_and_process_all_files(config_path, result_path)
                
                aggregated_batch_sizes.append(batch_size)
                aggregated_prefill_mean_latencies.append(prefill_mean_latency)
                aggregated_decode_mean_latencies.append(decode_mean_latency)
                aggregated_prefill_max_ram.append(prefill_max_ram)
                aggregated_decode_max_ram.append(decode_max_ram)
                aggregated_decode_throughput.append(decode_throughput)
                aggregated_prefill_throughput.append(prefill_throughput)
    
    return aggregated_batch_sizes, aggregated_prefill_mean_latencies, aggregated_decode_mean_latencies, aggregated_prefill_max_ram, aggregated_decode_max_ram, aggregated_decode_throughput, aggregated_prefill_throughput

### Plotting functions ###

def plot_decode_memory(batch_sizes, max_ram):    
    try:
        if batch_sizes and max_ram is not None:
            plot_data = pd.DataFrame({
                'Batch Size': batch_sizes,
                'Max RAM (MB)': max_ram
            })

            sns.set_theme(style='white')
            sns.barplot(data=plot_data, x='Batch Size', y='Max RAM (MB)',hue='Batch Size', palette="rocket")
            plt.title('Decode Max Memory per batch size')
            plt.xlabel('Batch Size')
            plt.ylabel('Max RAM (MB)')
    except ValueError as e:
        print(f"An error occured from missing values: {e}")
        return None

def plot_prefill_memory(batch_sizes, max_ram): 
    try:
        if batch_sizes and max_ram is not None:
            plot_data = pd.DataFrame({
                'Batch Size': batch_sizes,
                'Max RAM (MB)': max_ram
            })

            sns.set_theme(style='white')
            sns.barplot(data=plot_data, x='Batch Size', y='Max RAM (MB)',hue='Batch Size', palette="rocket")
            plt.title('Decode Max Memory per batch size')
            plt.xlabel('Batch Size')
            plt.ylabel('Max RAM (MB)')
    except ValueError as e:
        print(f"An error occured from missing values: {e}")
        return None
    

def plot_decode_latency(batch_sizes, mean_latencies):    
    try:
        if batch_sizes and mean_latencies is not None:
            plot_data = pd.DataFrame({
                'Batch Size': batch_sizes,
                'Mean Latency (s)': mean_latencies
            })

   
            sns.lineplot(data=plot_data, x='Batch Size', y='Mean Latency (s)', marker='o')
            plt.title('Decode Latency per Batch Size')
            plt.xlabel('Batch Size')
            plt.ylabel('Mean Latency (s)')
    except ValueError as e:
        print(f"An error occured from missing values: {e}")
        return None
    

def plot_prefill_latency(batch_sizes, mean_latencies):    
    try:
        if batch_sizes and mean_latencies is not None:
            plot_data = pd.DataFrame({
                'Batch Size': batch_sizes,
                'Mean Latency (s)': mean_latencies
            })

   
            sns.lineplot(data=plot_data, x='Batch Size', y='Mean Latency (s)', marker='o')
            plt.title('Prefill Latency per Batch Size')
            plt.xlabel('Batch Size')
            plt.ylabel('Mean Latency (s)')
    except ValueError as e:
        print(f"An error occured from missing values: {e}")
        return None
    

def plot_decode_throughput(batch_sizes, decode_throughputs):
    try:
        if batch_sizes and decode_throughputs is not None:            
            plot_data = pd.DataFrame({
                'Batch Size': batch_sizes,
                'Decode Throughput': decode_throughputs
            })
            sns.set_theme(style='white')
            sns.barplot(data=plot_data, x='Batch Size', y='Decode Throughput',hue='Batch Size', palette="rocket")
            plt.title('Decode throughput per batch size')
            plt.xlabel('Batch Size')
            plt.ylabel('Decode throughput')
    except ValueError as e:
        print(f"An error occured from missing values: {e}")
        return None
    

def plot_prefill_throughput(batch_sizes, prefill_throughputs):
    try:
        if batch_sizes and prefill_throughputs is not None:
            plot_data = pd.DataFrame({
                'Batch Size': batch_sizes,
                'Prefill Throughput': prefill_throughputs
            })
            sns.set_theme(style='white')
            sns.barplot(data=plot_data, x='Batch Size', y='Prefill Throughput',hue='Batch Size', palette="rocket")
            plt.title('Prefill Throughput per batch size')
            plt.xlabel('Batch Size')
            plt.ylabel('Prefill Throughput')
    except ValueError as e:
        print(f"An error occured from missing values: {e}")
        return None
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide the directory path as an argument.")
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Script to analyze torch trace file")
    parser.add_argument('--trace_file', type=str, required=True, help="trace file to be analyzed")
    parser.add_argument('--kernel_name', type=str, required=True, help="kernel name")
    parser.add_argument('--pic', action='store_true', default=False, help="Whether generate figure to show perf variance")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    file_name = args.trace_file
    kernel_name = args.kernel_name

    file_obj = open(file_name)
    obj = json.load(file_obj)
    trace_events = obj['traceEvents']
    kernel_time = []
    for e in trace_events:
        name = e["name"]
        if kernel_name in name:
            kt = e["dur"]
            kernel_time.append(kt)

    # np_time = np.array(kernel_time)
    kernel_time = kernel_time[10:]
    mean_time = round(np.mean(kernel_time), 2)
    max_time = round(np.max(kernel_time), 2)
    min_time = round(np.min(kernel_time), 2)
    print(f"Kernel {kernel_name} time statistics:")
    print(f"    Mean: {mean_time} us")
    print(f"    Max:  {max_time} us")
    print(f"    Min:  {min_time} us")
    print(f"    Num of calls: {len(kernel_time)}")

    if args.pic:
        plt.plot(kernel_time)
        plt.xlabel(f'{kernel_name} call in order')
        plt.ylabel(f'time (us)')
        plt.show()
        plt_name = f"plot_{file_name}_{kernel_name}.png"
        # replace '/' with '_'
        plt_name = plt_name.replace('/', '_')
        plt.savefig(plt_name)

if __name__ == "__main__":
    main()


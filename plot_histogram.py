import matplotlib.pyplot as plt
import os

import numpy as np


def main():
    """
    Reads token lengths from a text file and generates a histogram.
    """
    input_path = 'prompt_token_lengths.txt'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        print("Please run show_prompt_distro.py first to generate the data.")
        return


    with open(input_path, 'r') as f:
        prompt_lengths = [int(line.strip()) for line in f if line.strip()]

    print("Generating histogram...")
    plt.figure(figsize=(10, 6))
    plt.hist(prompt_lengths, bins=100, color='blue', alpha=0.7)
    plt.title('Distribution of Prompt Token Lengths')
    plt.xlabel('Prompt Length (tokens)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Calculate the 90th percentile
    percentile_90 = np.percentile(prompt_lengths, 90)

    print(f"The 90th percentile for prompt token length is: {int(percentile_90)}")
    print(f"Reading token lengths from {input_path}...")
    output_image_path = 'prompt_token_length_histogram.png'
    plt.savefig(output_image_path)
    print(f"Histogram saved to {output_image_path}")

if __name__ == "__main__":
    main()

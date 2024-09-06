import os
import csv
from collections import defaultdict


def extract_labels_from_txt(txt_directory):
    labels = set()
    label_counts = defaultdict(int) # Dictionary to count occurrences of each label

    # Iterate over all txt files in the directory
    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(txt_directory, filename)
            with open(file_path, 'r') as file:
                idx = 0
                for line in file:
                    if idx > 0: # ignore first line
                        # Split the line by whitespace and take the last element as the label
                        label = line.strip().split()[4:] # some labels are multiple words
                        label = " ".join(label) # some labels are multiple words so make resulting label one string
                        if label != "": # ignore empty strings
                            labels.add(label)
                            label_counts[label] += 1
                    idx += 1

    return sorted(labels), label_counts


def generate_class_labels_csv(labels, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["index", "mid", "display_name"])  # Write header

        for index, label in enumerate(labels):
            mid = f"/m/{label.replace(' ', '_')}"  # Create a mid with label replacing spaces with underscores
            display_name = label
            csvwriter.writerow([index, mid, display_name])


if __name__ == "__main__":
    # Set the path to your txt files directory and output CSV file
    txt_directory = "/home/jdcast/wav_training_data/TK/clean_label"
    output_csv = "/home/jdcast/Audio-Mamba-AuM/exps/tektite/data/class_labels_indices.csv"

    # Extract unique labels from .txt files
    labels, label_counts = extract_labels_from_txt(txt_directory)

    # Generate the class_labels_indices.csv file
    generate_class_labels_csv(labels, output_csv)

    # Print label counts
    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print(f"CSV file '{output_csv}' generated successfully.")
import os
import json
import random


def extract_labels_from_txt_file(txt_file):
    labels = set()
    with open(txt_file, 'r') as file:
        idx = 0
        for line in file:
            if idx > 0: # ignore first line
                # label = line.strip().split()[-1]
                label = line.strip().split()[4:]  # some labels are multiple words
                label = " ".join(label)  # some labels are multiple words so make resulting label one string
                if label != "":  # ignore empty strings
                    labels.add(f"/m/{label.replace(' ', '_')}")

            idx += 1
    return ','.join(sorted(labels))


def create_data_pairs(wav_dir, txt_dir):
    data_pairs = []

    for filename in os.listdir(wav_dir):
        if filename.endswith(".wav"):
            wav_file = os.path.join(wav_dir, filename)

            txt_file = os.path.join(txt_dir, filename.strip('.wav') + '_labels' + '.txt')
            # txt_file = os.path.join(txt_dir, filename.replace(".wav", ".txt"))

            if os.path.exists(txt_file):
                labels = extract_labels_from_txt_file(txt_file)
                data_pairs.append({
                    "wav": wav_file,
                    "labels": labels
                })

    return data_pairs


def split_data(data_pairs, train_ratio=0.8):
    random.shuffle(data_pairs)
    split_idx = int(len(data_pairs) * train_ratio)
    return data_pairs[:split_idx], data_pairs[split_idx:]


def write_json(data_pairs, output_json):
    with open(output_json, 'w') as json_file:
        json.dump({"data": data_pairs}, json_file, indent=2)


if __name__ == "__main__":
    # Define your directories for wav and txt files
    wav_dir = "/home/jdcast/wav_training_data/TK/wav"
    txt_dir = "/home/jdcast/wav_training_data/TK/clean_label"

    # Define output paths for train and test JSON files
    train_json = "/home/jdcast/Audio-Mamba-AuM/exps/tektite/data/train_data.json"
    test_json = "/home/jdcast/Audio-Mamba-AuM/exps/tektite/data/test_data.json"

    # Create data pairs
    data_pairs = create_data_pairs(wav_dir, txt_dir)

    # Split into train and test sets (80% train, 20% test)
    train_data, test_data = split_data(data_pairs, train_ratio=0.8)

    # Write JSON files
    write_json(train_data, train_json)
    write_json(test_data, test_json)

    print(f"Train data saved to {train_json}")
    print(f"Test data saved to {test_json}")
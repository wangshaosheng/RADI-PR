import json
from pathlib import Path


def load_training_data(input_file, target_file):
    with open(input_file, 'r') as f:
        inputs = f.readlines()
    with open(target_file, 'r') as f:
        targets = f.readlines()
    return list(zip(inputs, targets))


def extract_phrases(tokens, start_token, end_token):
    start_indices = [i for i, token in enumerate(tokens) if start_token in token]
    end_indices = [i for i, token in enumerate(tokens) if end_token in token]

    phrases = []
    while start_indices and end_indices:
        start = start_indices.pop(0)
        end = next((i for i in end_indices if i >= start), None)
        if end is not None:
            end_indices.remove(end)
            phrases.append((start, end))
    return phrases

def extract_key_changes(input_seq, target_seq):
    input_tokens = input_seq.split()
    target_tokens = target_seq.split()

    min_len = min(len(input_tokens), len(target_tokens))
    start, end = 0, min_len

    while start < min_len and input_tokens[start] == target_tokens[start]:
        start += 1
    while end > start and input_tokens[end - 1] == target_tokens[end - 1]:
        end -= 1

    error_phrase = " ".join(input_tokens[start:end]) if start < end else " ".join(input_tokens)
    correct_phrase = " ".join(target_tokens[start:end]) if start < end else " ".join(target_tokens)

    return error_phrase, correct_phrase


def generate_intervention_strategies(training_data):
    strategies = {}
    for input_seq, target_seq in training_data:
        input_tokens = input_seq.split()
        target_tokens = target_seq.split()

        error_phrases = extract_phrases(input_tokens, "<BUGS>", "<BUGE>")
        correct_phrases = extract_phrases(target_tokens, "<FIXS>", "<FIXE>")

        if error_phrases and correct_phrases:
            for (start, end) in error_phrases:
                error_phrase = " ".join(input_tokens[start:end + 1])
                for (c_start, c_end) in correct_phrases:
                    correct_phrase = " ".join(target_tokens[c_start:c_end + 1]).replace("<FIXS>", "").replace("<FIXE>", "")
                    strategies[error_phrase] = correct_phrase
                    break  
        else:
            error_phrase, correct_phrase = extract_key_changes(input_seq, target_seq)
            if error_phrase and correct_phrase:
                strategies[error_phrase] = correct_phrase

    print(f"Total strategies generated: {len(strategies)}")
    return strategies


def save_strategies_to_file(strategies, filepath):
    with open(filepath, 'w') as f:
        json.dump(strategies, f, indent=4)


if __name__ == "__main__":
    input_file = './text.train.input'  # Change to your file path
    target_file = './text.train.target'  # Change to your file path
    training_data = load_training_data(input_file, target_file)
    strategies = generate_intervention_strategies(training_data)
    save_strategies_to_file(strategies, 'intervention_strategies.json')

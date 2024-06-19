def truncate(sequences, max_length=50):
    """
    Truncates all sequences in a list to a maximum length.

    Args:
        sequences: A list of sequences (lists, strings, etc.).
        max_length: The maximum desired length for each sequence (default: 100).

    Returns:
        A new list containing the truncated sequences.
    """
    truncated_sequences = []
    for sequence in sequences:
        padding_length = max_length - len(sequence)
        if padding_length < 0:
            truncated_sequence = sequence[:max_length]
            truncated_sequences.append(truncated_sequence)
        else:
            padding_char = 'N'
            padded_sequence = sequence + padding_char * padding_length
            truncated_sequences.append(padded_sequence)
    
    return truncated_sequences

def read_domiknows_data(data_path):
    '''Load DNA sequence data from a file, convert to list of dictionaries and return train and test splits.'''
    data = []
    with open(data_path, 'r') as file:
        columns = file.readline().strip().split('\t')
        sum_length = 0
        cnt = 0
        for line in file:
            sequence, label = line.strip().split('\t')
            sum_length += len(sequence)
            cnt += 1

            data.append({'sequence': truncate(sequence, 100), 'label': int(label)})
        
        train = []
        test = []
        # print("data:", data)
        for i in range(0, len(data)):
            if i % 4 == 0:
                test.append(data[i])
            else:
                train.append(data[i])

    return train, test




# def split_data(sequences_tensor, labels_tensor, test_size=0.25):
#     X_train, X_test, y_train, y_test = train_test_split(sequences_tensor, labels_tensor, test_size=test_size, random_state=35)
#     return X_train, X_test, y_train, y_test

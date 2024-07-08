def truncate(sequence, max_length=50):
    """
    Truncates all sequences in a list to a maximum length.

    Args:
        sequences: A list of sequences (lists, strings, etc.).
        max_length: The maximum desired length for each sequence (default: 100).

    Returns:
        A new list containing the truncated sequences.
    """
    
    padding_length = max_length - len(sequence)
    if padding_length < 0:
        truncated_sequence = sequence[:max_length]
    else:
        padding_char = 'N'
        padded_sequence = sequence + padding_char * padding_length
        truncated_sequence = padded_sequence[:max_length]
   
    complementary_sequence = ''
    for nucleotide in truncated_sequence:
        if nucleotide == 'A':
            complementary_sequence += 'T'
        elif nucleotide == 'T':
            complementary_sequence += 'A'
        elif nucleotide == 'C':
            complementary_sequence += 'G'
        elif nucleotide == 'G':
            complementary_sequence += 'C'
        else:
            complementary_sequence += 'N'
    # print("complementary_sequences:", complementary_sequences)
    return [truncated_sequence, complementary_sequence]

def read_domiknows_data(data_path):
    '''Load DNA sequence data from a file, convert to list of dictionaries and return train and test splits.'''
    data = []
    with open(data_path, 'r') as file:
        columns = file.readline().strip().split('\t')
        sum_length = 0
        cnt = 0
        for line in file:
            
            sequence, label = line.strip().split('\t')
            
            # sum_length += len(sequence)
            # cnt += 1

            data.append({'strand': truncate(sequence, 10), 'label': int(label)-1})
            
        
        train = []
        test = []
        # print("data:", data)
        for i in range(0, len(data)):
            if i % 4 == 0:
                test.append(data[i])
            else:
                train.append(data[i])

    # print("train:", train[:10])
    # print("test:", test[:10])

    return train[:10], test[:10]




# def split_data(sequences_tensor, labels_tensor, test_size=0.25):
#     X_train, X_test, y_train, y_test = train_test_split(sequences_tensor, labels_tensor, test_size=test_size, random_state=35)
#     return X_train, X_test, y_train, y_test

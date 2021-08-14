def data_generation():
    import numpy as np
    import random
    import keras

    train_labels = []
    train_samples = []

    test_labels = []
    test_samples = []

    # Training sets: random generate 100K Sequences of 60 letters
    for i in range(100000):
        train_sample = ""
        for j in range(60):
            train_sample += (random.choice('ACGT'))
        train_samples.append(train_sample)
    print("len(train_samples): ", len(train_samples))

    # Testsets
    for i in range(200):
        test_sample = ""
        for j in range(60):
            test_sample += (random.choice('ACGT'))
        test_samples.append(test_sample)
    print("len(test_samples): ", len(test_samples))

    # For trainingsets. Check each sequence, mark the one
    # Self-defined rule: exact 1 TTT
    for seq in train_samples:
        if seq.count("TTT") == 2:
            train_labels.append(1)
        else:
            train_labels.append(0)
    print("len(train_labels): ", len(train_labels))

    # For testsets
    for seq in test_samples:
        if seq.count("TTT") == 2:
            test_labels.append(1)
        else:
            test_labels.append(0)
    print("len(test_labels): ", len(test_labels))  


    # one-hot encoding
    # index: A=0, C=1, G=2, T=3 
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    train_samples_onehot = []

    for seq in train_samples:
        train_sample_onehot = np.array([], dtype=int).reshape(0,4)
        for char in seq:
            bp=np.array([0, 0, 0, 0])
            bp[mapping[char]] = 1
            train_sample_onehot = np.concatenate((train_sample_onehot, [bp]), axis=0)
        train_samples_onehot.append(train_sample_onehot)

    train_samples = train_samples_onehot

    print("train_samples.shape: ", np.array(train_samples).shape)


    test_samples_onehot = []
    for seq in test_samples:
        test_sample_onehot = np.array([], dtype=int).reshape(0,4)
        for char in seq:
            bp=np.array([0, 0, 0, 0])
            bp[mapping[char]] = 1
            test_sample_onehot = np.concatenate((test_sample_onehot, [bp]), axis=0)
        test_samples_onehot.append(test_sample_onehot)

    test_samples = test_samples_onehot

    print("test_samples.shape: ", np.array(test_samples).shape)


    # Dataset to numpy array, dimension transform, and one-hot encoding
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    train_labels = keras.utils.to_categorical(train_labels, num_classes=2)

    test_samples = np.array(test_samples)
    test_labels = np.array(test_labels)

    return train_samples, train_labels, test_samples, test_labels
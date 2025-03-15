import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import pickle
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)  # Compute log probabilities
        self.loss = nn.NLLLoss()  # Cross-entropy loss

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden_layer = self.W1(input_vector)  # Apply first linear transformation
        hidden_layer = self.activation(hidden_layer)  # Apply ReLU activation
        hidden_layer = self.dropout(hidden_layer)  # Apply dropout
        output_layer = self.W2(hidden_layer)  # Apply second linear transformation
        predicted_vector = self.softmax(output_layer)  # Convert to probability distribution
        return predicted_vector

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))

    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="test.json", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    model = FFNN(len(word2index), args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Add L2 regularization
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    TRAIN_ACCR = []
    TRAIN_LOSS = []
    VALIDATION_ACCR = []

    # Add a patience parameter
    patience = 10  # Allow validation accuracy to decrease for 3 epochs before stopping
    no_improvement_count = 0  # Counter for epochs without improvement

    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        print(f"Training started for epoch {epoch + 1}")
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                output = model(input_vector.unsqueeze(0))
                example_loss = model.compute_Loss(output, torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

                TRAIN_LOSS.append(example_loss)

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

        trainning_accuracy = correct / total
        train_loss = loss_total / loss_count
        TRAIN_ACCR.append(trainning_accuracy)
        TRAIN_LOSS.append(train_loss)

        print(f"Loss after epoch {epoch + 1}: {loss_total / loss_count}")
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print(f"Validation started for epoch {epoch + 1}")

        for input_vector, gold_label in tqdm(valid_data):
            output = model(input_vector.unsqueeze(0))
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        validation_accuracy = correct / total
        print(f"Validation accuracy for epoch {epoch + 1}: {validation_accuracy}")
  
        VALIDATION_ACCR.append(validation_accuracy)  # Move this line here
        
        # Early stopping condition
        if validation_accuracy < last_validation_accuracy:
            no_improvement_count += 1
            print(f"No improvement in validation accuracy for {no_improvement_count} epochs.")
        else:
            no_improvement_count = 0  # Reset the counter if validation accuracy improves

        if no_improvement_count >= patience:
            stopping_condition = True
            print(f"Stopping early due to no improvement in validation accuracy for {patience} epochs.")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1

    if args.test_data:
        print(f"========== Loading test data from: {args.test_data} ==========")
        test_data, _ = load_data(args.test_data, args.val_data)
        test_data = convert_to_vector_representation(test_data, word2index)

        model.eval()
        correct = 0
        total = 0
        start_time = time.time()
        random.shuffle(test_data)
        print(f"Testing started.")

        for input_vector, gold_label in tqdm(test_data):
            output = model(input_vector.unsqueeze(0))
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        print(f"Testing completed.")
        print(f"Test accuracy: {correct / total}")
        print(f"Testing time: {time.time() - start_time}")

        output_filename = "test_FFNN_results.txt"
        with open(output_filename, "w") as f:
            f.write(f"Test accuracy: {correct / total}\n")
            f.write(f"Testing time: {time.time() - start_time}\n")
            f.write(f"=====Below is for plotting=====\nTraining Accr: {TRAIN_ACCR}\n")
            f.write(f"Training Loss: {TRAIN_LOSS}\n")
            f.write(f"Validation Accuracies: {VALIDATION_ACCR}")


        print(f"Test results saved to {output_filename}")

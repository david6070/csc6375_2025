import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.LSTM(input_dim, h, self.numOfLayer, batch_first=True)  # Changed to LSTM
        self.dropout = nn.Dropout(p=0.5)  # Added dropout for regularization
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        _, (hidden, _) = self.rnn(inputs)  # Obtain hidden state (LSTM)
        hidden = self.dropout(hidden[-1])  # Apply dropout
        output_layer = self.W(hidden)  # Apply linear transformation
        predicted_vector = self.softmax(output_layer)  # Convert to probability distribution
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]-1)))
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
    model = RNN(50, args.hidden_dim)  # Model with 50 input dimension
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lowered learning rate
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    TRAIN_ACCR = []
    TRAIN_LOSS = []

    

    for epoch in range(args.epochs):
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
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                vectors = [word_embedding.get(i.lower(), word_embedding['unk']) for i in input_words]
                vectors = torch.tensor(np.array(vectors), dtype=torch.float32).view(1, len(vectors), -1)  # Adjusted shape

                output = model(vectors)

                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding.get(i.lower(), word_embedding['unk']) for i in input_words]
            vectors = torch.tensor(np.array(vectors), dtype=torch.float32).view(1, len(vectors), -1)

            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        validation_accuracy = correct / total
        print(f"Validation accuracy for epoch {epoch + 1}: {correct / total}")
VALIDATION_ACCR = []  # Add this line to initialize the list

for epoch in range(args.epochs):
    ...
    VALIDATION_ACCR.append(validation_accuracy)  # Now this will work
    print(f"Validation accuracy for epoch {validation_accuracy}")

    # After your validation phase (inside the main block)
    if args.test_data:

        # Load test data from test.json
        print(f"========== Loading test data from: {args.test_data} ==========")
        test_data, _ = load_data(args.test_data, args.val_data)  # No need for validation data in testing phase
        
        if not test_data:
            print("Test data loading failed. Exiting...")
            exit(1)
        
        # Test phase after training
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        start_time = time.time()
        random.shuffle(test_data)
        print(f"Testing started.")
        
        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding.get(i.lower(), word_embedding['unk']) for i in input_words]
            vectors = torch.tensor(np.array(vectors), dtype=torch.float32).view(1, len(vectors), -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        print(f"Testing completed.")
        print(f"Test accuracy: {correct / total}")
        print(f"Testing time: {time.time() - start_time}")

        # Optionally save the outputs
        output_filename = "test_rrn_results.txt"
        with open(output_filename, "w") as f:
            f.write(f"Test accuracy: {correct / total}\n")
            f.write(f"Testing time: {time.time() - start_time}\n")
            f.write(f"=====Below is for plotting=====\nTraining Accr: {TRAIN_ACCR}\n")
            f.write(f"Training Loss: {TRAIN_LOSS}\n")
            f.write(f"Validation Accuracies: {VALIDATION_ACCR}\n")

        print(f"Test results saved to {output_filename}")

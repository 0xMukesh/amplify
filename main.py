import torch
import torch.nn as nn
import torch.nn.functional as F
import random

BLOCK_SIZE = 3
VOCAB_SIZE = 27
N_EMBEDDING = 10
N_HIDDEN = 200

MAX_STEPS = 200000
BATCH_SIZE = 32


def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * BLOCK_SIZE

        for ch in list(w) + ["."]:
            X.append(context)
            Y.append(stoi[ch])

            context = context[1:] + [stoi[ch]]

    return torch.tensor(X), torch.tensor(Y)


@torch.no_grad()
def calculate_loss(split):
    x, y = {
        "train": (X_train, Y_train),
        "test": (X_test, Y_test)
    }[split]

    embedding = C[x].view(-1, BLOCK_SIZE * N_EMBEDDING)
    res1 = (embedding @ W1 + b1).tanh()
    res2 = (res1 @ W2 + b2)

    loss = F.cross_entropy(res2, y)
    return loss.item()


words = open("names.txt", "r").read().splitlines()
chars = sorted(set("".join(words)))

stoi = {s: i+1 for i, s in enumerate(chars)}
stoi["."] = 0

itos = {i: s for s, i in stoi.items()}

n1 = round(0.8 * len(words))

random.shuffle(words)

X_train, Y_train = build_dataset(words[:n1])
X_test, Y_test = build_dataset(words[n1:])

# take in three characters into the context and embed them into a 2 dimensional space
# and pass all the three characters together into the neural network
C = torch.randn((VOCAB_SIZE, N_EMBEDDING))

# layer 1
# 6 to 100 neurons, fully connected layer and uses tanh as the activation function
W1 = torch.randn((N_EMBEDDING * BLOCK_SIZE, N_HIDDEN))
b1 = torch.randn(N_HIDDEN) * 0.01

W1 = nn.init.kaiming_normal_(W1, mode="fan_in", nonlinearity="tanh")

# layer 2
# 100 to 27 neurons, fully connected layer and uses softmax as the activation function
W2 = torch.randn((N_HIDDEN, VOCAB_SIZE)) * 0.01
b2 = torch.zeros(VOCAB_SIZE)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True


for i in range(MAX_STEPS):
    # constructing a mini batch
    # generates a tensor of size 32 elements with random numbers from [0, N)
    ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))

    # forward pass
    embedding = C[X_train[ix]].view(-1, N_EMBEDDING * BLOCK_SIZE)
    pre_res1 = embedding @ W1 + b1
    res1 = (embedding @ W1 + b1).tanh()

    res2 = (res1 @ W2 + b2)

    # calculating the loss using cross entropy as the log function
    loss = F.cross_entropy(res2, Y_train[ix])
    loss.grad = None

    print(f"iter no {i + 1} - loss = {loss}")

    for p in parameters:
        p.grad = torch.zeros_like(p)

    loss.backward()

    lr = 0.1 if i < 100000 else 0.01

    for p in parameters:
        if p.grad is not None:
            p.data += -lr * p.grad


training_loss = calculate_loss("train")
testing_loss = calculate_loss("test")

print(f"training loss - {training_loss}\ntesting loss - {testing_loss}")

for _ in range(20):
    out = []
    context = [0] * BLOCK_SIZE

    while True:
        embedding = C[torch.tensor([context])].view(
            1, BLOCK_SIZE * N_EMBEDDING)  # (1, 3, 10) -> (1, 30)

        res1 = torch.tanh(embedding @ W1 + b1)
        res2 = res1 @ W2 + b2

        probs = F.softmax(res2, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()

        if ix == 0:
            break

        context = context[1:] + [ix]
        out.append(itos[int(ix)])

    print("".join(out))

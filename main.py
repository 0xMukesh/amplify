from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn.functional as F
import random

BLOCK_SIZE = 3
VOCAB_SIZE = 27
N_EMBEDDING = 10
N_HIDDEN = 200

MAX_STEPS = 200000
BATCH_SIZE = 32


class Layer(ABC):
    def __init__(self):
        self.training = True

    @abstractmethod
    def __call__(self, x) -> torch.Tensor:
        pass

    @abstractmethod
    def parameters(self) -> List[torch.Tensor]:
        pass


class Linear(Layer):
    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias

        return self.out

    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]


class BatchNorm(Layer):
    def __init__(self, dim, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.bn_mean = torch.zeros(dim)
        self.bn_std = torch.ones(dim)
        # self.training = True

    def __call__(self, x):
        if self.training:
            mean = x.mean(0, keepdim=True)
            std = x.std(0, keepdim=True)
        else:
            mean = self.bn_mean
            std = self.bn_std

        xhat = (x - mean)/(std)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.bn_mean = (1 - self.momentum) * \
                    self.bn_mean + self.momentum * mean
                self.bn_std = (1 - self.momentum) * \
                    self.bn_std + self.momentum * std

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding(Layer):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class Sequential(Layer):
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers

    def __call__(self, x):
        self.out = x
        for layer in self.layers:
            self.out = layer(self.out)
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def eval(self):
        for layer in self.layers:
            layer.training = False

    def train(self):
        for layer in self.layers:
            layer.training = True


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

    model.eval()
    x = model(x)
    model.train()

    loss = F.cross_entropy(x, y)
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

model = Sequential([
    Embedding(VOCAB_SIZE, N_EMBEDDING),
    Flatten(),
    Linear(N_EMBEDDING * BLOCK_SIZE, N_HIDDEN),
    Tanh(),
    BatchNorm((1, N_HIDDEN)),
    Linear(N_HIDDEN, VOCAB_SIZE),
])

parameters = model.parameters()

for p in parameters:
    p.requires_grad = True


for i in range(MAX_STEPS):
    ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))

    # forward pass
    x = model(X_train[ix])

    # calculating loss
    loss = F.cross_entropy(x, Y_train[ix])
    loss.grad = None

    print(f"iter no {i + 1} - loss = {loss}")

    # setting gradients to zero
    for p in parameters:
        p.grad = torch.zeros_like(p)

    # backpropagation
    loss.backward()

    # optimization (SGD)
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
        x = torch.tensor([context])

        model.eval()
        x = model(x)
        model.train()

        probs = F.softmax(x, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()

        if ix == 0:
            break

        context = context[1:] + [ix]
        out.append(itos[int(ix)])

    print("".join(out))

import torch

words = open("names.txt", "r").read().splitlines()

chars = sorted(set("".join(words)))

stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0

itos = {i: s for s, i in stoi.items()}

xs, ys = [], []

for word in words:
    for ch1, ch2 in zip(word[0], word[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]

        xs.append(idx1)
        ys.append(idx2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = torch.nn.functional.one_hot(xs, 27).float()
W = torch.randn((27, 27), requires_grad=True)

for i in range(100):
    # forward pass
    logits = (xenc @ W)
    counts = logits.exp()
    prob = counts/counts.sum(dim=1, keepdim=True)

    # calculating loss for the entire dataset
    loss = -prob[torch.arange(ys.nelement()),
                 ys].log().mean() + (0.01) * (W**2).mean()

    print(f"iter {i+1}: loss = {loss}")

    # setting the gradient to be equal to 0
    W.grad = torch.zeros((27, 27))
    # backward pass
    loss.backward()
    # updating the weights
    W.data += -30 * W.grad

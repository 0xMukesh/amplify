import torch

words = open("names.txt", "r").read().splitlines()

chars = sorted(set("".join(words)))

stoi = {s: i+1 for i, s in enumerate(chars)}
stoi["."] = 0

itos = {i: s for s, i in stoi.items()}

xs, ys = [], []

for word in words:
    chs = ["."] + list(word) + ["."]

    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        idx3 = stoi[ch3]

        xs.append((idx1, idx2))
        ys.append(idx3)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = torch.nn.functional.one_hot(xs, 27).float()  # Nx2x27
xenc = xenc.view(-1, 2*27)
W = torch.randn((2*27, 27), requires_grad=True)

for i in range(1000):
    logits = (xenc @ W)
    counts = logits.exp()
    prob = counts/counts.sum(dim=1, keepdim=True)

    loss = -prob[torch.arange(ys.nelement()),
                 ys].log().mean() + (0.01) * (W**2).mean()

    print(f"iter {i+1}: loss = {loss}")

    W.grad = torch.zeros((2*27, 27))
    loss.backward()
    W.data += -3 * W.grad

for i in range(10):
    out = [".", "."]

    while True:
        idx1 = stoi[out[-2]]
        idx2 = stoi[out[-1]]

        xenc = torch.nn.functional.one_hot(
            torch.tensor([idx1, idx2], dtype=torch.long), num_classes=27).float()
        xenc = xenc.view(-1, 2*27)

        logits = xenc @ W
        counts = logits.exp()
        prob = counts/counts.sum(dim=1, keepdim=True)

        ix = torch.multinomial(prob, num_samples=1, replacement=True).item()

        out.append(itos[int(ix)])

        if ix == 0:
            break

    print("".join(out))

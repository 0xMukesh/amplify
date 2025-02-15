import torch

words = open("names.txt", "r").read().splitlines()

# list of all the chars in the dataset
chars = sorted(list(set(''.join(words))))
# char to int mapping
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi["."] = 0

itos = {i: s for s, i in stoi.items()}

# 2d model representing the entire dataset
# dataset[row, col] represent the frequency of that pair
# `row` and `col` are the integer mappings of the chars
dataset = torch.zeros(27, 27, dtype=torch.float32)

for word in words:
    bigram = ["."] + list(word) + ["."]

    # converts the word into pairs and increments the freq in the dataset tensor
    for ch1, ch2 in zip(bigram, bigram[1:]):
        dataset[stoi[ch1], stoi[ch2]] += 1


for i in range(30):
    row = 0
    out = []

    while (True):
        # convert the freqs into probabilties
        p = dataset[row]/dataset[row].sum()
        # get predications based on the above probabilities
        idx = torch.multinomial(p, 1, True)
        chstr = itos[int(idx.item())]

        # loop until special character is reached i.e. end of the string is reached
        if (chstr == "."):
            break

        out.append(chstr)
        row = int(idx)

    print("".join(out))

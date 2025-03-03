# notes

## bigram model

1. in bigram model, all the words are split up into pairs of two letters (bigrams) (emma -> (., e), (e, m), (m, m), (m, a), (a, .))
2. a matrix is formed which contains the frequency of each pair
3. the matrix is then normalized to get the probability distributions
4. to predict a word, a loop is ran which starts with `.` character and then samples the next character using `torch.multinomial` function which just returns the next character according to the probability distributions
5. after getting the next character, same thing is done again but with the next character's row and the loop runs until `.` character is hit again.
6. negative log likelihood is used as the loss function
7. a random constant is added to all the counts in the matrix which acts like model smoothing/regularization

## neural network

1. all the words are split up into pairs of two letters and the first element of that pair is appended to `xs` and the second element is appended to the `ys` array. `xs` array is passed to the network and `ys` are the targets
2. `xs` array is transformed by applying one-hot encoding
3. a random weight matrix is initialized using normal distribution and forward pass is performed
4. negative-log likelihood is again used as the loss function and along with it L2 loss used for regularization

   ```py
   loss = -prob[torch.arange(ys.nelement()), ys].log().mean() + (0.01) * (W**2).mean()
   ```

5. backward pass is performed and the weights are updated using basic SGD

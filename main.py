#from matplotlib import pyplot as plt
import torch

def print_stats(_N):
    # sum each row to get the number of times each token appears in the corpus
    sum_N = _N.sum(dim=1)
    # sort the values in descending order, retaining only the frequency, not the token ids
    sorted_values = sum_N.sort()[0]  

    # print some statistics
    print(f'Number of singular tokens: {(sum_N == 1).sum().item()}')
    print(f'Count of twenty most common tokens: {sorted_values[-20:].tolist()}')

    # commented out - table is too sparse for pyplot to show anything
    # plt.figure(figsize=(16,16))
    # plt.imshow(torch.log(N + 1), cmap='Blues')
    # plt.show()


# create a list of all chorales
with open('all-chorales-SB-start-of-beat.txt', 'r') as file:
    chorales = (file.read()
        .splitlines())

    # tokens will hold an item for each unique token
    file.seek(0)
    all_tokens = file.read().split()
    vocabulary = set(all_tokens)
    print(f"All tokens: {len(all_tokens)}")
    print(f"Unique tokens: {len(vocabulary)} ({len(vocabulary) / len(all_tokens) * 100:.2f}%)")

# change tokens from a set to a list and sort it alphabetically
# the index of a token into this list will be the token identifier
vocabulary = sorted(list(vocabulary))
vocabulary.remove('.') # we want to ensure the identifer for '.' is 0
stoi = {s:i+1 for i, s in enumerate(vocabulary)}
stoi['.'] = 0

# create a reverse lookup table
itos = {i:s for s, i in stoi.items()}

# create a tensor to hold the number of times each token in the row is followed by the token in the column
# we removed '.' from the vocabulary so we must add 1 to the count
N = torch.zeros((len(vocabulary) + 1, len(vocabulary) + 1), dtype=torch.int32)

for c in chorales:
    # split chorale into tokens
    tokens = c.split()
    # zip tokens and tokens[1:] to create tuples of each token and its successor
    for t1, t2 in zip(tokens, tokens[1:]):

        # increment the associated entry in N
        ix1 = stoi[t1]
        ix2 = stoi[t2]
        N[ix1, ix2] += 1

print_stats(N)

# # compute the probability of each token's being followed by another token
# p = N / N.sum(dim=1, keepdim=True)

# # the multinomial function samples values from a list representing a probability distribution
# # it returns the index into the list at the specified probability
# # for example, if p = [0.1, 0.9] and we sample 10 times, we will get, on average, 9 ones and 1 zero
# g = torch.Generator().manual_seed(2147483647)

# # generate a chorale:
# ix = 0
# while True:
#     ix = torch.multinomial(p[ix], 1, replacement=True, generator=g).item()
#     print(itos[ix])
#     if ix == 0:
#         break


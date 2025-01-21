import torch

# create a list of all chorales
with open('all-chorales-chords-noeom.txt', 'r') as file:
    chorales = (file.read()
        .splitlines())

    # tokens will hold an item for each unique token
    file.seek(0)
    all_tokens = file.read().split()
    vocabulary = set(all_tokens)
    print(f"All tokens: {len(all_tokens)}")
    print(f"Unique tokens: {len(vocabulary)}")

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

sum_N = N.sum(dim=1)
sorted_values = sum_N.sort()[0]  # sort the values

print(f'Number of singular tokens: {(sum_N == 1).sum().item()}')
print(f'Counf of twenty most common tokens: {sorted_values[-20:].tolist()}')



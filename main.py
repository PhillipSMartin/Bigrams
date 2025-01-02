import torch, sys

# create a list of all chorales
with open('all-chorales-clean.txt', 'r') as file:
    # chorale will hold an item for each chorale
    chorales = file.read().splitlines()

    # tokens will hold an item for each unique token
    # [SOC] and [EOC] will both be changed to "."
    file.seek(0)
    tokens = set(file.read()
                .replace('\n', ' ')
                .replace("[SOC]",".")
                .replace("[EOC]",".")
                .split())

print(f"We have {len(tokens)} tokens in our vocabulary.")

# change tokens from a set to a list and sort it alphabetically
# the index of a token into this list will be the token identifier
tokens = sorted(list(tokens))
print(tokens[:10])

# create a tensor to hold the number of times each token is followed by a particular token
torch.zeros((len(tokens), len(tokens)), dtype=torch.int32)

# create a t with count for each bigram
b = {}
for c in chorales:
    # split chorale into tokens
    tokens = c.split()
    # zip tokens and tokens[1:] to create tuples of each token and its successor
    for t1, t2 in zip(tokens, tokens[1:]):
        # use the tuple as a key in the dictionary and count occurences
        bigram = (t1, t2)
        b[bigram] = b.get(bigram, 0) + 1

sb = sorted(b.items(), key = lambda kv: -kv[1])
print ( sb[:10] )

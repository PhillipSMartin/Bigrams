import matplotlib
matplotlib.use('TkAgg')
import matplotlib.colors
import matplotlib.pyplot as plt
import torch

# create a list of all chorales
with open('all-chorales-clean.txt', 'r') as file:
    # [SOC] and [EOC] will both be changed to "."; [EOM] will be changed to "/"
    chorales = (file.read()
        .replace("[SOC]",".")
        .replace("[EOC]",".")
        .replace("[EOM]","/")
        .splitlines())

    # tokens will hold an item for each unique token
    # [SOC] and [EOC] will both be changed to "."; [EOM] will be changed to "/"
    file.seek(0)
    vocabulary = set(file.read()
                .replace('\n', ' ')
                .replace("[SOC]",".")
                .replace("[EOC]",".")
                .replace("[EOM]","/")
                .split())

# change tokens from a set to a list and sort it alphabetically
# the index of a token into this list will be the token identifier
vocabulary = sorted(list(vocabulary))
stoi = {s:i for i, s in enumerate(vocabulary)}

# create a tensor to hold the number of times each token in the row is followed by the token in the column
N = torch.zeros((len(vocabulary), len(vocabulary)), dtype=torch.int32)

for c in chorales:
    # split chorale into tokens
    tokens = c.split()
    # zip tokens and tokens[1:] to create tuples of each token and its successor
    for t1, t2 in zip(tokens, tokens[1:]):

        # increment the associated entry in N
        ix1 = stoi[t1]
        ix2 = stoi[t2]
        N[ix1, ix2] += 1

# show N via matplotlib
# must run sudo apt-get install python3-tk to run outside a Jupyter notebook
# need logarithmic scale to see anything
plt.imshow(N, norm=matplotlib.colors.LogNorm())
plt.show()




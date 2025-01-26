#from matplotlib import pyplot as plt
import torch

def print_stats(N):
    """ 
        Takes a tensor of bigram counts and prints some statistics 
    """

    # sum each row to get the number of times each token appears in the corpus
    sum_N = N.sum(dim=1)
    # sort the values in descending order, retaining only the frequency, not the token ids
    sorted_values = sum_N.sort()[0]  

    # print some statistics
    print(f'Number of singular tokens: {(sum_N == 1).sum().item()}')
    print(f'Count of twenty most common tokens: {sorted_values[-20:].tolist()}')

    # commented out - table is too sparse for pyplot to show anything
    # plt.figure(figsize=(16,16))
    # plt.imshow(torch.log(N + 1), cmap='Blues')
    # plt.show()

def get_fileName():
    """ 
        Asks the user questions to determine the name of the encoding file to use.

        Inputs:
            None
        Outputs:
            name of the file
    """

    sb = ''
    encoding = 0
    while sb != 'y' and sb != 'n':
        sb = input("Soprano-bass (y) or full chords (n)? ").lower()
    while encoding not in [1, 2, 3]:
        encoding = int(input("Encoding for each change (1), each beat (2), or start of beat (3)? "))
    file_name = "all-chorales-" + ("SB-" if sb == 'y' else "chords-")
    if encoding == 1:
        file_name += "noeom.txt"
    elif encoding == 2:
        file_name += "per-beat.txt"
    else:
        file_name += "start-of-beat.txt"
    return file_name
    
def build_vocabulary(file_name, verbose=True):
    """
        Builds a list of chorales and a vocabulary (a list of distinct tokens,
            sorted but with '.' (start-or-end token) as the first token).

        Inputs:
            file-name of a file containing one line for each chorale
            boolean indicating whether to print statistics
        Outputs: 
            list chorales
            vocabulary
    """
    chorales = []
    vocabulary = set()

    # create a list of all chorales
    with open(file_name, 'r') as file:
        for line in file:
            # Store the full line in chorales
            chorales.append(line.strip())
            # Add tokens to vocabulary
            vocabulary.update(line.split())

    if verbose:
        all_tokens_count = sum(len(chorale.split()) for chorale in chorales)
        print(f"Total number of tokens in the corpus: {all_tokens_count}")
        print(f"Number of unique tokens in the corpus: {len(vocabulary)} ({len(vocabulary) / all_tokens_count * 100:.2f}%)")

    # change vocabulary from a set to a list and sort it alphabetically
    vocabulary = sorted(list(vocabulary))

    # remove '.' and  add it to the beginning of the list
    vocabulary.remove('.')
    vocabulary.insert(0, '.')

    # return the list of chorales and the list of tokens
    return chorales, vocabulary

def build_N(chorales, vocabulary, stoi):
    """
        Create a tensor to hold the number of times each token in the row is followed by the token 
            in the column.

        Inputs:
            list of chorales
            vocabulary
            a dictionary mapping tokens to their ids
        Outputs:
            tensor of bigram counts
    """
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

    return N

def generate_chorale(P, itos, g):
    """
        Generate a chorale by sampling from the probability distribution

        Inputs:
            probability each token in the row is followed by the token in the column
            dictionary mapping ids to tokens
            random number generator

        Outputs:
            a list of tokens comprising a chorale
    """
    ix = 0
    chorale = []
    while True:
        # the multinomial function samples values from a list representing a probability distribution
        # it returns the index into the list at the specified probability
        # for example, if p = [0.1, 0.9] and we sample 10 times, we will get, on average, 9 ones and 1 zero
        ix = torch.multinomial(P[ix], 1, replacement=True, generator=g).item()
        chorale.append(itos[ix])
        if ix == 0:
            return chorale



# ------------- main code --------------

# read the chorales and build vocabulary
chorales, vocabulary = build_vocabulary(get_fileName())

# create a table to convert from token to id (the index into vocabulary)
stoi = {s:i for i, s in enumerate(vocabulary)}
# create a reverse lookup table
itos = {i:s for s, i in stoi.items()}

# build the bigram count matrix
N = build_N(chorales, vocabulary, stoi)
print_stats(N)

# compute the probability of each token's being followed by another token
# dim = 1 means we are summing each column producing an n x 1 tensor 
#  (if keepdim is False (the default), the summed-up dimension is squeezed out producing a n-element tensor)
#  The division is possible because of broadcasting rules:
#    Each dimension must be equal or one must be a 1 or one must not exist
P = N / N.sum(dim=1, keepdim=True)

# generate a chorale
g = torch.Generator().manual_seed(2147483647)
chorale = generate_chorale(P, itos, g) 

for token in chorale:
    print(token)


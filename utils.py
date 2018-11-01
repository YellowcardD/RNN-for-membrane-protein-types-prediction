import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical


# 1461088e9d9d2d57c211d7f4c134d6dc_234.pssm  11
# 3cdf43ae0c6c68e94fb01e122fe0d93f_376.pssm  12
# 2920d1b8faf6e02bf96bbf63b28e3307_312.pssm  2
# c7f0d845cab5bf4b20ac633215590d16_24.pssm  3
# c7f0d845cab5bf4b20ac633215590d16_68.pssm  4
# f77a8f2b82abdf12ecf47a71ad4efbf1_303.pssm  51
# 9ba4630649cce0c94add6a7f7805b6f4_380.pssm  52
# 34b97b3e710517ecffa64f1357215de9_345.pssm  53
# d2e5a220fa68c262cf2ccad6ad4d8c99_288.pssm  54
# 3d4bd507c40bc0c0d76a969c399021d0_151.pssm  6
# 043214218cbd7e98eac4783876032930_182.pssm  7
# f01419925764272d1bd8d51475a8d798_275.pssm   81
# e7baacb591d01562a62e163637b29a2f_335.pssm   82

train_data_dir_name = {"1461088e9d9d2d57c211d7f4c134d6dc_%d.pssm": 234 , "3cdf43ae0c6c68e94fb01e122fe0d93f_%d.pssm": 376 ,
                 "2920d1b8faf6e02bf96bbf63b28e3307_%d.pssm": 312 , "c7f0d845cab5bf4b20ac633215590d16_%d.pssm": 68 ,
                 "f77a8f2b82abdf12ecf47a71ad4efbf1_%d.pssm": 303,"9ba4630649cce0c94add6a7f7805b6f4_%d.pssm": 380,
                 "34b97b3e710517ecffa64f1357215de9_%d.pssm": 345, "d2e5a220fa68c262cf2ccad6ad4d8c99_%d.pssm": 288,
                 "3d4bd507c40bc0c0d76a969c399021d0_%d.pssm": 151 , "043214218cbd7e98eac4783876032930_%d.pssm": 182,
                 "f01419925764272d1bd8d51475a8d798_%d.pssm": 275, "e7baacb591d01562a62e163637b29a2f_%d.pssm": 335}
cnt = 0
max_len = 5000
X = []
protein_len = []

def preprocessing_one_protein(data):
    """
    get PSSM for one protein and pad the protein whose length less than max_len
    :return: a numpy array for one protein with shape 5000x20
    """
    matrix = np.zeros((5000, 20))
    index = 0
    for str in data:
        a = str[0].split(' ')
        a = a[6:]
        array = [float(x) for x in a if x != '']
        array = np.array(array[:20])
        matrix[index,:] = array
        index = index + 1

    protein_len.append(index+1)
    return matrix

# Get all PSSM for every and stored in a numpy array X
for name, num in train_data_dir_name.items():
    for i in range(num):
        data = pd.read_csv(("data/" + name %(i+1)), sep='\t',skiprows=2)
        data = data[0:-5].values
        x = preprocessing_one_protein(data)
        X.append(x)

# Save X to file
np.save('protein_len.npy', protein_len)
np.save('X.npy', X)

# 40c04cb7c746dc0fa5d56c06b8894d94_444.pssm 1
# c211b546f1411fb2ca932e644382a16f_78.pssm 2
# c211b546f1411fb2ca932e644382a16f_84.pssm 3
# c211b546f1411fb2ca932e644382a16f_96.pssm 4
# 1ccfab131c3de8b0516c353ba4bdaf28_210.pssm 51
# 5dfd1ff70e09f0b64d2cc38e9bcc0dfb_143.pssm 52
# 224e30584c0f2f6331c3f97f4c439a59_188.pssm 53
# eceb34a8983f5ad80deb833c1bd3e129_179.pssm 54
# d862b29f3a96fde259752e83393c5ef6_212.pssm 55
# 08a11bf4215b8dae32745317da4d05d6_247.pssm 56
# 8490a6cdd0e81b39ada99cc6c6cc5215_363.pssm 57
# 5aa4c04616dce71b6845995ab596c2a6_186.pssm 58
# 551e1aab7da1578345ad12a5c9cf98bb_253.pssm 59
# 9d2a7617a0d4dc7e5f2267c9f60ceeb5_313.pssm 510
# 65345b297595542bc336385cea4914d0_333.pssm 511
# 599649c5763ad33b4b9a685480d6a2bb_157.pssm 512
# 6a92434d4cfd00f7a97ad90f8db07398_259.pssm 513
# 391d543644243106face942e2b69db12_221.pssm 514
# e4195302823c3583c3a77e02fe9bfedf_1.pssm 515
# 92bc55b9c988c0574bc29270e91279e0_38.pssm 6
# 92bc55b9c988c0574bc29270e91279e0_84.pssm 7
# c6a4ee734416fc5a9080a88635a74b65_444.pssm 8

test_data_dir_name = {"40c04cb7c746dc0fa5d56c06b8894d94_%d.pssm": 444, "c211b546f1411fb2ca932e644382a16f_%d.pssm": 96,
                      "1ccfab131c3de8b0516c353ba4bdaf28_%d.pssm": 210, "5dfd1ff70e09f0b64d2cc38e9bcc0dfb_%d.pssm": 143,
                      "224e30584c0f2f6331c3f97f4c439a59_%d.pssm": 188, "eceb34a8983f5ad80deb833c1bd3e129_%d.pssm": 179,
                      "d862b29f3a96fde259752e83393c5ef6_%d.pssm": 212, "08a11bf4215b8dae32745317da4d05d6_%d.pssm": 247,
                      "8490a6cdd0e81b39ada99cc6c6cc5215_%d.pssm": 363, "5aa4c04616dce71b6845995ab596c2a6_%d.pssm": 186,
                      "551e1aab7da1578345ad12a5c9cf98bb_%d.pssm": 253, "9d2a7617a0d4dc7e5f2267c9f60ceeb5_%d.pssm": 313,
                      "65345b297595542bc336385cea4914d0_%d.pssm": 333, "599649c5763ad33b4b9a685480d6a2bb_%d.pssm": 157,
                      "6a92434d4cfd00f7a97ad90f8db07398_%d.pssm": 259, "391d543644243106face942e2b69db12_%d.pssm": 221,
                      "e4195302823c3583c3a77e02fe9bfedf_%d.pssm": 1, "92bc55b9c988c0574bc29270e91279e0_%d.pssm": 84,
                      "c6a4ee734416fc5a9080a88635a74b65_%d.pssm": 444}


for name, num in test_data_dir_name.items():
    for i in range(num):
        data = pd.read_csv(("test_data/" + name %(i+1)), sep='\t',skiprows=2)
        data = data[0:-5].values
        x = preprocessing_one_protein(data)
        X.append(x)

X = np.array(X)
np.save('protein_len_test.npy', protein_len)
np.save('X_test.npy', X)
print(X.shape)

def get_dict():
    
    char = sorted(['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H'])
    char_to_index = {}
    index_to_char = {}
    char_to_vec = {}
    index = 0
    for c in char:
        char_to_index[c] = index
        index_to_char[index] = c
        char_to_vec[c] = to_categorical(index, num_classes=20)
        index = index + 1

    return char_to_index, index_to_char, char_to_vec

def processing_one_sequence(data):

    matrix = np.zeros((5000, 20))
    index = 0
    for str in data:
        a = str[0][6]
        matrix[index, :] = char_to_vec[a]
        index = index + 1

    return matrix


# char_to_index, index_to_char, char_to_vec = get_dict()
# print(char_to_index)
# print(to_categorical(0, 20), to_categorical(1, 20))

# for name, num in train_data_dir_name.items():
#     for i in range(num):
#         data = pd.read_csv(("data/" + name %(i+1)), sep='\t',skiprows=2)
#         data = data[0:-5].values
#         x = processing_one_sequence(data)
#         X.append(x)
#
# np.save('X_train_oh.npy', X)

# for name, num in test_data_dir_name.items():
#     for i in range(num):
#         data = pd.read_csv(("test_data/" + name %(i+1)), sep='\t',skiprows=2)
#         data = data[0:-5].values
#         x = processing_one_sequence(data)
#         X.append(x)
#
# np.save('X_test_oh.npy', X)

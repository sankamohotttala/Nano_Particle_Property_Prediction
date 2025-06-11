from itertools import permutations, combinations
import random
import copy

required_number = 100

layer_node_config = [256,128,64,32,16,8,4,2]
tmp_combin = list(combinations(layer_node_config, 4))
# tmp_permu = list(permutations(tmp_combin[0]))

all_permu = []
for combin in tmp_combin:
    all_permu.extend(list(permutations(combin)))

# pyramid architecture small to large
# inverted pyramid architecture
# other architectures
pyramid_archi = []
inverted_archi = []
other_archi = []

for perm_ in all_permu:
    tmp_changing = perm_[:]
    
    perm = list(perm_[:])
    asending_sort = list(tmp_changing[:])
    asending_sort.sort()

    descending_sort = list(tmp_changing[:])
    descending_sort.sort(reverse = True)

    if perm == asending_sort: # pyramid archi
        pyramid_archi.append([perm,'pyramid'])
        pass
    elif perm == descending_sort:
        inverted_archi.append([perm,'inv_pyramid'])
    else: # other mixed structures
        other_archi.append([perm,'other'])

all_architectures = {'0':pyramid_archi, '1':inverted_archi,'2':other_archi}
# all_architectures_copy = dict(all_architectures)
all_architectures_copy_2 = copy.deepcopy(all_architectures) # we keep this without doing any changes for later usage

numbers = [0, 1, 2] #pyramdi, inverted_pyramid , other
weights = [0.4, 0.4, 0.2]

#method 1 - for all permutations giving equal weight
randomlist = random.sample(range(len(all_permu)), required_number)
layer_architectures = [all_permu[val] for val in randomlist]

#method 2 - with the above weight list values
selecte_architetures = []
for i in range(required_number):
    tmp_random_val = random.choices(numbers, weights)[0]
    tmp_permutations = all_architectures[str(tmp_random_val)]
    random_index = random.sample(range(len(tmp_permutations)), 1)[0]
    selected_archi =  tmp_permutations.pop(random_index)
    selecte_architetures.append(selected_archi)

    
    # tmp_permutations[random_index]





    #contune from this
    # make sure we are not resampling the permutations
    # original lists should be there for reference - use a copy to remove the selected one
# value_counts = Counter(tmp__)


a=1



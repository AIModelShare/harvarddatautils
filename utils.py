import numpy as np

def concat_list_of_dict(ld, keys):
    # handles index (which is a list of strings)
    return {key: np.concatenate([np.asarray(d[key]) for d in ld]) for key in keys}

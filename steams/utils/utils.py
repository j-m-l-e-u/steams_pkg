import random
import string
import numpy as np

def get_random_alphanumeric_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return(result_str.lower())

def encode_cos(values, max_val):
    res = np.cos(2 * np.pi * values/max_val)
    return res

def encode_sin(values, max_val):
    res = np.sin(2 * np.pi * values/max_val)
    return res

import torch

import time

do_time_prof = False

def time_record(list_name):
    def inner_func(func):
        def time_func(*args, **kwargs):
            torch.cuda.synchronize()
            time_start = time.time()
            ans = func(*args, **kwargs)
            torch.cuda.synchronize()
            list_name.append(round((time.time()-time_start)*1000, 2))
            return ans
        return time_func
    return inner_func
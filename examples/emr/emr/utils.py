def seed(s=1, deterministic=True):
    import os
    import random
    import numpy as np
    import torch

    os.environ['PYTHONHASHSEED'] = str(s)  # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)  # this function will call torch.cuda.manual_seed_all(s) also

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

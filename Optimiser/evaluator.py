import numpy as np
from config import get_config
def f(x,lambda_coef=0.3):
    return -np.sum((x**2 -1)**2 - lambda_coef*(x-1)**2 + 5)


cfg = get_config()
N = cfg["env"]["state_dim"]
m1=np.array([-0.5]*N)
m2=np.array([0.5]*N)

def f_continous(x):
    return -np.log(np.sum((x-m1)**2)+0.00001)-np.log(np.sum((x-m2)**2)+0.01)
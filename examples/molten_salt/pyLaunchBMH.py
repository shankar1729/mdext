import os
from unicodedata import decimal
import numpy as np

# nohup python ../pyLaunch.py > out &
# nohup bash ../nacl_bmh.job > out &


# Loop over U in gaussian potential and launch jobs

s = 1. # sigma width gaussian
T = 1300. # kelvin
P = 1. # pressue  XXXX overridden later to be none TODO clean up
p = 2 # atom type to apply the potential to (1-based)
g = 'planar'


endRange = 5.0
stepSize = 0.5

# Ui=-0.4
# sweep through potential amplitudes

    

for Ui in np.around(np.arange(0,endRange*2 + stepSize ,stepSize), decimals=1)-endRange:  
    print(f"{Ui:+.1f}")
    o = f"test-U{Ui:+.1f}.h"
    log = o[:-3]+"_out"

    os.system(f"sbatch ../nacl_bmh.job {Ui} {s} {T} {P} {p} {g} {o}")
    # os.system(f"bash ../nacl_bmh.job {Ui} {s} {T} {P} {p} {g} {o} > {log} &")  
    # break
   

    
    
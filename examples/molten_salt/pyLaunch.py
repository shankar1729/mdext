import os
from unicodedata import decimal
import numpy as np

# nohup python ../pyLaunch.py > out &
# nohup bash ../nacl_bmh.job > out &


# Loop over U in gaussian potential and launch jobs

s = 1. # sigma width gaussian
T = 1300. # kelvin
P = 1. # pressue
p = 1 # atom type to apply the potential to (1-based)
g = 'planar'

endRange = 0.2
stepSize = 0.1

# Ui=-0.4
# sweep through potential amplitudes

for Ui in np.around(np.arange(0,endRange*2 + stepSize ,stepSize), decimals=1)-endRange:  
    print(f"{Ui:.1f}")
    o = f"NaCl_U{Ui:.1f}.h5"
    log = o[:-3]+"_out"
    
    # os.system(f"sbatch ../nacl_bmh.job {Ui} {s} {T} {P} {p} {g} {o}")
    os.system(f"bash ../nacl_bmh.job {Ui} {s} {T} {P} {p} {g} {o}")  #  > {log} &
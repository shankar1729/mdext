import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import os
import matplotlib as mpl

endRange = 5.0
stepSize = 0.5

particle = 1

plt.figure(1)
# --- initialize colormap to color by V0
normalize = mpl.colors.Normalize(vmin=-endRange, vmax=endRange)
cmap = mpl.cm.get_cmap("RdBu")

for Ui in np.around(np.arange(0,endRange*2 + stepSize ,stepSize), decimals=1)-endRange:  
    print(f"{Ui:+.1f}")
    
    with h5py.File(f"test-U{Ui:+.1f}.h5", "r") as fp:
        r = np.array(fp["r"])
        n = np.array(fp["n"])
        V = np.array(fp["V"])
    plt.plot(r,n[:,particle]/(1.),color=cmap(normalize(Ui)), lw=1)
    plt.xlabel("z [$\AA$]",fontsize=11,fontname="Times New Roman")
    plt.ylabel("$n_{H}(z)$",fontsize=11,fontname="Times New Roman")
    # plt.ylabel("$n_{H}(z)/n_{bulk}$",fontsize=11,fontname="Times New Roman")
    
    # plt.legend()
    plt.legend( prop={
            'family': 'Times New Roman', "size": 7, 'stretch': 'normal'})    # plt.title('Classical, -0.4 eV',fontsize=12,fontname="Times New Roman")
    plt.xticks(fontsize=11,fontname="Times New Roman")
    plt.yticks(fontsize=11,fontname="Times New Roman")

# plt.plot(r, V, color="k", lw=1, ls="dashed")  # plot largest shaped V
plt.xlim([r.min(),r.max()])
# plt.ylim([0,4])

# --- add colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])
plt.colorbar(sm, label=r"Perturbation strength, $\lambda$")
figure = plt.gcf()
# figure.set_size_inches(3.35, 2.2)
plt.savefig('plotMany.pdf', dpi=600, bbox_inches='tight')
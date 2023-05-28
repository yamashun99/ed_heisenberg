import numpy as np
import matplotlib.pyplot as plt

L = 6
Gijs = np.load('Gijs.npy')
xs = [i for i in range(L)]

omegamax = 5
omegamesh = 20
omegas = [iomega / omegamesh * omegamax for iomega in range(omegamesh)]

fig = plt.figure()
ax = fig.add_subplot()
mappable = ax.pcolor(xs, omegas, Gijs.real)
fig.colorbar(mappable)
fig.savefig('Gij_real.pdf')

fig = plt.figure()
ax = fig.add_subplot()
mappable = ax.pcolor(xs, omegas, Gijs.imag)
fig.colorbar(mappable)
fig.savefig('Gij_imag.pdf')

print(Gijs)

import matplotlib.pyplot as plt
import numpy as np

def gaussian(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi*sigma**2) * np.exp(-(x - mu) ** 2 / sigma ** 2)

x = np.linspace(-5, 5, 1000)
mu = 0
sigma = 1
y_target = gaussian(x, 0, 1)
y_behavior = gaussian(x, -4.4, 0.2)
y_behavior_2 = gaussian(x, -0.5, 0.6)


fig = plt.figure(figsize=(12,6))

np.random.seed(0)
bins = np.linspace(-3, 3, 20)
n = 10
y_rand = np.random.normal(mu, sigma, n)
y_rand = y_rand[y_rand>0]
print(y_rand)
plt.hist(y_rand, bins=bins, density=True, alpha=0.5, label=r'Empirical policy $\pi_{\mathcal{D}}$')

plt.plot(x,y_target, label='Target policy $\pi_{\\theta}(a|s)$', linewidth=2)
plt.plot(x,y_behavior, label='PROPS Behavior policy $\pi_{\phi}(a|s)$ without regularization', linewidth=2)
plt.plot(x,y_behavior_2, label='PROPS Behavior policy $\pi_{\phi}(a|s)$ with regularization', linewidth=2)


plt.xlabel(rf'a', fontsize=16)
plt.ylabel(rf'$\pi_\phi(a|s)$', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax = fig.axes[0]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])

plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=18)
plt.tight_layout()
plt.savefig('figures/regularization.png')
plt.show()

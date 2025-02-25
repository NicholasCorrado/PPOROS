import matplotlib.pyplot as plt
import numpy as np

def gaussian(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi*sigma**2) * np.exp(-(x - mu) ** 2 / sigma ** 2)

x = np.linspace(-1, 1, 1000)
mu = 0
sigma = 0.2
y_target = gaussian(x, mu, sigma)
y_ros = gaussian(x, -0.5, sigma/4)
y_props = gaussian(x, 0.15, sigma/2)


# empirical data
np.random.seed(42)
bins = np.linspace(-1, 1, 30)
n = 5
y_data = np.concatenate([[-0.1, -0.25, 1]])
loss_ros = -np.log(gaussian(y_data, mu, sigma)).sum()
loss_props = -gaussian(y_data, mu, sigma).sum()
# print(mu - loss)
print(np.log(gaussian(y_data, mu, sigma)).sum())
print(np.log(gaussian(y_data[-1], mu, sigma)).sum())


fig = plt.figure(figsize=(6,6))
plt.hist(y_data, bins=bins, density=True, alpha=0.5, label=r'Empirical policy $\pi_{\mathcal{D}}$')
plt.plot(x,y_target, label='Target policy $\pi_{\\theta}(a|s)$', linewidth=2,)
# plt.plot(x,y_ros, label='Behavior policy $\pi_{\phi}(a|s) + $regularization', linewidth=2)

y_ideal = gaussian(x, 0.15, 0.1)
plt.plot(x,y_ros, label='ROS Behavior Policy $\pi_{\phi}(a|s)$', linewidth=2,)
plt.plot(x,y_props, label='PROPS Behavior Policy $\pi_{\phi}(a|s)$', linewidth=2)

# plt.plot(x,y_behavior_2, label='Behavior policy $\pi_{\phi}(a|s) + $regularization', linewidth=2)
plt.xticks([-1, -0.5, 0, 0.5, 1])

plt.xlabel(rf'a', fontsize=16)
plt.ylabel(rf'$\pi_\phi(a|s)$', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax = fig.axes[0]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])

plt.legend(bbox_to_anchor=(0.42, 1), loc='lower center', fontsize=18)
plt.tight_layout()
plt.savefig('figures/destructive_updates.png')
plt.show()

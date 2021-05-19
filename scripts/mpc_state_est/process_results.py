from os import walk
from os.path import realpath, dirname, splitext, join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

res_dir = join(dirname(realpath(__file__)), "_results")

RES = {}
STATES = {}
OBS = {}

for (dirpath, dirnames, filenames) in walk(res_dir):
    for filename in filenames:
        name, ext = splitext(filename)
        if ext == ".npy":
            print(name)
            if "state" in name:
                print(name)
                states = np.load(join(res_dir, filename))
                config = "_".join(name.split("_")[1:4])
                if config in STATES:
                    STATES[config].append(states)
                else:
                    STATES[config] = [states]
            elif "obs" in name:
                obs = np.load(join(res_dir, filename))
                config = "_".join(name.split("_")[1:4])
                if config in OBS:
                    OBS[config].append(obs)
                else:
                    OBS[config] = [obs]
            else:
                cost = np.load(join(res_dir, filename))
                config = "_".join(name.split("_")[:3])
                if config in RES:
                    RES[config].append(cost)
                else:
                    RES[config] = [cost]

# quick illustration
W = 600 / 30
H = 400 / 30
T = 100
z_traj = np.zeros((T, 6))
z_traj[:, 0] = np.linspace(W / 4, 3 * W / 4, T)
z_traj[:, 1] = H / 2 + (H / 4) * np.sin(np.linspace(0, 2 * np.pi, T))
z_traj[:, 2] = 2 * np.pi * np.heaviside(np.linspace(-1, 1, T), 1)
print(z_traj)
print(STATES.keys(), OBS.keys())
i2c_states = STATES["i2c_FF_high"][0]
ilqr_states = STATES["iLQR_FB_high"][0]
i2c_obs = OBS["i2c_FF_high"][0]
ilqr_obs = OBS["iLQR_FB_high"][0]
f, ax = plt.subplots(2, 2)
ax[0, 0].set_title("Flight Trajectory")
ax[0, 1].set_title("Control Sequence")
ax[1, 0].set_ylabel("i2c")
ax[0, 0].set_ylabel("iLQR")
for i, (state, obs) in enumerate(zip([ilqr_states, i2c_states], [ilqr_obs, i2c_obs])):
    a = ax[i, 0]
    a.plot(z_traj[:, 0], z_traj[:, 1], "k--")
    a.plot(state[:, 0], state[:, 1], "m-")
    for t in range(obs.shape[0]):
        a.plot(obs[t, [0, 2]], obs[t, [1, 3]], "y")
    a.set_ylim(0, 13)
    a.set_xlim(0, 20)

    a = ax[i, 1]
    a.plot(state[:, 6], "c-", label="$u_1$")
    a.plot(state[:, 7], "b-", label="$u_2$")

    matplotlib2tikz.save(join(res_dir, "mpc_compare.tex"))

for config, costs in RES.items():
    print(
        config,
        len(costs),
        np.percentile(np.asarray(costs), [10, 90]),
        min(costs),
        max(costs),
    )

from DKBM import DSKinematicBicycleModel
from DKBM_casadi import csDSKBM
from scripts import *
from controls.MPC import MPC

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import casadi as cs
 
N_STATES = 4
N_ACTIONS = 3
L = 0.3
l_f = 0.1
l_r = 0.2
T = 10          # MPC horizon
N = 50          # Control Interval
DT = 0.2        # dt = T/N

MAX_SPEED = 1.5
MAX_STEER = np.radians(30)
MAX_D_ACC = 1.0
MAX_D_STEER = np.radians(30)  # rad/s
MAX_ACC = 1.0
REF_VEL = 1.0

cs_kbm = csDSKBM(N_STATES, N_ACTIONS, L, l_f, l_r, T, N)
kbm = DSKinematicBicycleModel(N_STATES, N_ACTIONS, L, l_f, l_r, T, DT)

# Step 1: Create a sample track
track = generate_path_from_wp(
    [0, 3, 4, 6, 10, 12, 14, 6, 1, 0],
    [0, 0, 2, 4, 3, 3, -2, -6, -2, -2], 0.05
)

sim_duration = 200  # time steps
opt_time = []

# VARIABLES FOR TRACKING
x_sim = np.zeros((N_STATES, sim_duration))
u_sim = np.zeros((N_ACTIONS, sim_duration - 1))

# Step 2: Create starting conditions x0
x_sim[:, 0] = np.array([0.0, -0.25, 0.0, np.radians(0)]).T

# Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
u_bar_start = np.zeros((N_ACTIONS, T))
u_bar_start[0, :] = MAX_ACC / 2
u_bar_start[1, :] = 0.0
u_bar_start[2, :] = 0.0

l_a = []
l_df = []
l_dr = []
l_state = []

MPC_PARAMS = {
    "n_x": N_STATES,
    "m_u": N_ACTIONS,
    "T": T,
    "dt": 0.2,
    "X_lb": cs.DM([-cs.inf, -cs.inf, 0, -cs.inf],),
    "X_ub": cs.DM([cs.inf, cs.inf, MAX_SPEED, cs.inf],),
    "U_lb": cs.DM([-MAX_ACC, -MAX_STEER, -MAX_STEER]), 
    "U_ub": cs.DM([MAX_ACC, MAX_STEER, MAX_STEER]),
    "dU_b": cs.DM([MAX_D_ACC, MAX_D_STEER, MAX_D_STEER]),
    "Q": cs.DM(np.diag([20, 20, 10, 0])),
    "Qf": cs.DM(np.diag([30, 30, 30, 0])),
    "R": cs.DM(np.diag([10, 10, 10])),
    "R_": cs.DM(np.diag([10, 10, 10]))
}

mpc = MPC(MPC_PARAMS, cs_kbm)

for sim_time in range(sim_duration - 1):
    iter_start = time.time()
    
    # Re-optimizing up to five times with the previous solved controls as a warm-start to try to get a better solution.
    for i_iter in range(1):
        # For the first iteration we use dummy controls as u_bar
        if i_iter == 0:
            u_bar = u_bar_start
        
        X_mpc, U_mpc = mpc.predict(x_sim[:, sim_time], u_bar, track)
        
        a_mpc   = np.array(U_mpc[0, :]).flatten()
        d_f_mpc = np.array(U_mpc[1, :]).flatten()
        d_r_mpc = np.array(U_mpc[2, :]).flatten()
        u_bar_new = np.vstack((a_mpc, d_f_mpc, d_r_mpc))

        delta_u = np.sum(np.sum(np.abs(u_bar_new - u_bar), axis=0), axis=0)
        if delta_u < 0.05:
            break
            
        u_bar = u_bar_new
    
    current_state = X_mpc[:, 0]
    l_state.append(current_state)
    x_mpc       = np.array(X_mpc[0, :]).flatten()
    y_mpc       = np.array(X_mpc[1, :]).flatten()
    v_mpc       = np.array(X_mpc[2, :]).flatten()
    theta_mpc   = np.array(X_mpc[3, :]).flatten()

    a_mpc   = np.array(U_mpc[0, :]).flatten()
    df_mpc  = np.array(U_mpc[1, :]).flatten()
    dr_mpc  = np.array(U_mpc[2, :]).flatten()

    l_a.append(a_mpc[0])
    l_df.append(df_mpc[0])
    l_dr.append(dr_mpc[0])

    u_bar = np.vstack((a_mpc, df_mpc, dr_mpc))
    
    # Take first action
    u_sim[:, sim_time] = u_bar[:, 0]
    
    # Measure elpased time to get results from cvxpy
    opt_time.append(time.time() - iter_start)

    # move simulation to t+1
    # tspan = [0, DT]
    x_sim[:, sim_time + 1] = cs_kbm.forward_one_step(x_sim[:, sim_time], u_sim[:, sim_time])
    # x_sim[:, sim_time + 1] = kbm.forward_one_step(x_sim[:, sim_time], u_sim[:, sim_time])

print("TOTAL TIME: {}".format(np.sum(opt_time)))

# 12.10 seconds
# 18.933314323425293 (previous implementation)

# Visualization
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(track[0, :], track[1, :], "b")
ax1.scatter(x_sim[0, :], x_sim[1, :], s=0.5, color='red', zorder=1)
# for i in range(x_sim.shape[1]):
#     ax1.text(
#         x_sim[0, i], x_sim[1, i], str(i),
#         fontsize=4, color='black', ha='center', va='center',
#         zorder=2  # Make sure it's above the scatter dots
#     )

ax1.plot(x_sim[0, :], x_sim[1, :], color='green', zorder=-1)
# plt.plot(x_traj[0, :], x_traj[1, :])
ax1.axis("equal")
ax1.set_ylabel("y")
ax1.set_xlabel("x")

ax2 = fig.add_subplot(gs[1, 0]) 
x = np.arange(len(l_a))
ax2.plot(x, l_a)
ax2.set_ylabel("acceleration command")
ax2.set_xlabel("time")

ax3 = fig.add_subplot(gs[1, 1])
x = np.arange(len(l_df))
ax3.plot(x, l_df)
ax3.set_ylabel("Front steering")
ax3.set_xlabel("time")

ax4 = fig.add_subplot(gs[1, 2])
x = np.arange(len(l_dr))
ax4.plot(x, l_dr)
ax4.set_ylabel("Rear steering")
ax4.set_xlabel("time")
plt.show()

# Test
# cs_kbm.integrate(np.random.rand(4, 1), np.random.rand(3, 1))
# cs_kbm.RK4(np.random.rand(4, 1), np.random.rand(3, 1), np.random.rand(4, 4), np.random.rand(4, 3))

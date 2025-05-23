import matplotlib.pyplot as plt
import numpy as np


class MPCPlotter:
    def __init__(self, track, sim_duration):
        # Initialize figure and create a grid with 2 rows, 3 columns
        self.fig = plt.figure(figsize=(10, 8))
        gs = self.fig.add_gridspec(2, 3)

        self.ax_track = self.fig.add_subplot(gs[0, :])
        self.ax_track.axis("equal")
        self.ax_track.set_xlabel("x (m)")
        self.ax_track.set_ylabel("y (m)")
        
        self.ax_acc = self.fig.add_subplot(gs[1, 0])
        self.ax_acc.set_xlabel("Time")
        self.ax_acc.set_ylabel("Acceleration (m/s^2)")
        
        self.ax_df = self.fig.add_subplot(gs[1, 1])
        self.ax_df.set_ylabel("Front Steering (radians)")
        self.ax_df.set_xlabel("Time")
        
        self.ax_dr = self.fig.add_subplot(gs[1, 2])
        self.ax_dr.set_xlabel("Time")
        self.ax_dr.set_ylabel("Rear Steering (radians)")

        if len(track) != 0:
            self.track = track
            self.plot_track(track)

        # For storing incoming data
        self.X = np.zeros((4, sim_duration))
        self.U = np.zeros((3, sim_duration))
        
        self.current_timestep = 0
    
    def plot_track(self, track):
        self.ax_track.plot(track[0, :], track[1, :], "b", label="Track")

    def plot_new_data(self, c_state, actions, l_nn_idx, i_timestep):
        """
        Params:
            c_state: cartesian state (x, y) of where robot currently is
            actions: only accepts actions of following form (acceleration, front steering, rear steering)
        """
        # Updates incoming data
        self.X[:, i_timestep] = c_state
        self.U[:, i_timestep] = actions
        
        N_timesteps = np.arange(i_timestep)

        self.ax_track.clear()
        self.plot_track(self.track)
    
        
        # We now have all the indices [nn_idx, ...]
        horizon = self.track[:, l_nn_idx]
        
        self.ax_track.scatter(horizon[0, :], horizon[1, :], s=0.8, color='orange', zorder=1)
        self.ax_track.scatter(self.X[0, :i_timestep], self.X[1, :i_timestep], s=0.5, color='red', zorder=0.5)
        self.ax_track.plot(self.X[0, :i_timestep], self.X[1, :i_timestep], color='green', zorder=-1)

        self.ax_acc.clear()
        self.ax_acc.plot(N_timesteps, self.U[0, :i_timestep])
        self.ax_acc.set_ylabel("accel (m/s)")
        self.ax_acc.set_xlabel("Time")

        self.ax_df.clear()
        self.ax_df.plot(N_timesteps, self.U[1, :i_timestep])
        self.ax_df.set_ylabel("Front Steering (rad)")
        self.ax_df.set_xlabel("Time")

        self.ax_dr.clear()
        self.ax_dr.plot(N_timesteps, self.U[2, :i_timestep])
        self.ax_dr.set_ylabel("Rear Steering (rad)")
        self.ax_dr.set_xlabel("Time")

    def plot_new_data_without_track(self, c_state, actions, i_timestep):
        """
        Params:
            c_state: cartesian state (x, y) of where robot currently is
            actions: only accepts actions of following form (acceleration, front steering, rear steering)
        """
        # Updates incoming data
        self.X[:, i_timestep] = c_state
        self.U[:, i_timestep] = actions
        
        N_timesteps = np.arange(i_timestep)

        self.ax_track.scatter(self.X[0, :i_timestep], self.X[1, :i_timestep], s=0.5, color='red', zorder=0.5)
        self.ax_track.plot(self.X[0, :i_timestep], self.X[1, :i_timestep], color='green', zorder=-1)

        self.ax_acc.clear()
        self.ax_acc.plot(N_timesteps, self.U[0, :i_timestep], color='r')
        self.ax_acc.set_ylabel("accel (m/s)")
        self.ax_acc.set_xlabel("Time")

        self.ax_df.clear()
        self.ax_df.plot(N_timesteps, self.U[1, :i_timestep], color='g')
        self.ax_df.set_ylabel("Front Steering (rad)")
        self.ax_df.set_xlabel("Time")

        self.ax_dr.clear()
        self.ax_dr.plot(N_timesteps, self.U[2, :i_timestep], color='b')
        self.ax_dr.set_ylabel("Rear Steering (rad)")
        self.ax_dr.set_xlabel("Time")

    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
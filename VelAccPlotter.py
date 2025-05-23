import matplotlib.pyplot as plt
import numpy as np


class VelAccPlotter:
    def __init__(self, sim_duration):
        # Initialize figure and create a grid with 2 rows, 3 columns
        self.fig = plt.figure(figsize=(10, 8))
        gs = self.fig.add_gridspec(2, 3)

        self.ax_track = self.fig.add_subplot(gs[0, :])
        self.ax_track.axis("equal")
        self.ax_track.set_xlabel("x (m)")
        self.ax_track.set_ylabel("y (m)")
        
        self.ax_v_wheel = self.fig.add_subplot(gs[1, 0])
        self.ax_v_wheel.set_xlabel("Time")
        self.ax_v_wheel.set_ylabel("Wheel Velocity Command (m/s)")
        
        self.ax_v_com = self.fig.add_subplot(gs[1, 1])
        self.ax_v_com.set_ylabel("COM Velocity (m/s)")
        self.ax_v_com.set_xlabel("Time")
        
        self.ax_a_com = self.fig.add_subplot(gs[1, 2])
        self.ax_a_com.set_ylabel("COM Acceleration (m/s^2)")
        self.ax_a_com.set_xlabel("Time")

        # For storing incoming data
        # [Wheel Velocity Command, COM Velocity, COM Acceleration]
        self.data = np.zeros((3, sim_duration))
        self.X = np.zeros((4, sim_duration))
            
        self.current_timestep = 0
    

    def plot_new_data(self, c_state, vel_acc_data, i_timestep):
        """
        Params:
            c_state: cartesian state (x, y) of where robot currently is
            vel_acc_data: only accepts data of following form [wheel velocity, com velocity, com accel]
        """
        # Updates incoming data
        self.X[:, i_timestep] = c_state
        self.data[:, i_timestep] = vel_acc_data
        
        N_timesteps = np.arange(i_timestep)

        self.ax_track.scatter(self.X[0, :i_timestep], self.X[1, :i_timestep], s=0.5, color='red', zorder=0.5)
        self.ax_track.plot(self.X[0, :i_timestep], self.X[1, :i_timestep], color='green', zorder=-1)

        self.ax_v_wheel.clear()
        self.ax_v_wheel.plot(N_timesteps, self.data[0, :i_timestep], color='r')
        self.ax_v_wheel.set_xlabel("Time")
        self.ax_v_wheel.set_ylabel("Wheel Velocity Command (m/s)")

        self.ax_v_com.clear()
        self.ax_v_com.plot(N_timesteps, self.data[1, :i_timestep], color='g')
        self.ax_v_com.set_ylabel("COM Velocity (m/s)")
        self.ax_v_com.set_xlabel("Time")

        self.ax_a_com.clear()
        self.ax_a_com.plot(N_timesteps, self.data[2, :i_timestep], color='b')
        self.ax_a_com.set_ylabel("COM Acceleration (m/s^2)")
        self.ax_a_com.set_xlabel("Time")

    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
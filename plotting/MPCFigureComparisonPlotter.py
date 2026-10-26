import matplotlib.pyplot as plt
import numpy as np


class MPCFigureComparisonPlotter:
    """Thesis figure: two runs on one track (paths on top, the three MPC actions below)."""
    def __init__(self, track, model_1_name, model_2_name, track_name):
        # Initialize figure and create a grid with 2 rows, 3 columns
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 3)

        self.ax_track = self.fig.add_subplot(gs[0, :])
        self.ax_acc = self.fig.add_subplot(gs[1, 0])
        self.ax_df = self.fig.add_subplot(gs[1, 1])        
        self.ax_dr = self.fig.add_subplot(gs[1, 2])
        
        self.model_1_name = model_1_name
        self.model_2_name = model_2_name
        self.track_name = track_name

        self.track = track
        self.plot_track(track)
    
    def setup_axes(self):
        self.ax_track.axis("equal")
        self.ax_track.set_xlabel("x (m)")
        self.ax_track.set_ylabel("y (m)")
        self.ax_track.set_title(f"{self.model_1_name} vs {self.model_2_name} on Track {self.track_name}")
        self.ax_track.legend()

        self.ax_acc.set_xlabel("Timestep (0.1s)")
        self.ax_acc.set_ylabel("Acceleration ($m/s^2$)")
        self.ax_acc.legend()

        self.ax_df.set_ylabel("Front Steering (radians)")
        self.ax_df.set_xlabel("Timestep (0.1s)")
        self.ax_df.legend()

        self.ax_dr.set_xlabel("Timestep (0.1s)")
        self.ax_dr.set_ylabel("Rear Steering (radians)")
        self.ax_dr.legend()

    def clear_axes(self):
        self.ax_track.clear()
        self.ax_acc.clear()
        self.ax_df.clear()
        self.ax_dr.clear()

    def plot_track(self, track):
        self.ax_track.plot(track[0, :], track[1, :], "b", label="Track")

    def plot_full_data(self, x_1, u_1, x_2, u_2):
        """x_*: (4, n + 1) states, u_*: (3, n) actions of each run. The runs may have different lengths."""

        self.clear_axes()

        self.plot_track(self.track)

        # Plotting robot 1's trajectory
        self.ax_track.scatter(
            x_1[0, :],
            x_1[1, :],
            s=1.0,
            alpha=0.9,
            color='forestgreen',
            zorder=0.5,
            # label=f"Trajectory per timestep using {self.model_1_name}"
        )
        self.ax_track.plot(
            x_1[0, :],
            x_1[1, :],
            color='forestgreen',
            alpha=0.9,
            zorder=-1,
            label=f"Trajectory using {self.model_1_name}"
        )

        # Plotting robot 2's trajectory
        self.ax_track.scatter(
            x_2[0, :],
            x_2[1, :],
            s=1.0,
            alpha=0.9,
            color='crimson',
            zorder=0.5,
            # label=f"Trajectory per timestep using {self.model_2_name}"
        )
        
        self.ax_track.plot(
            x_2[0, :],
            x_2[1, :],
            color='crimson',
            alpha=0.9,
            zorder=-1,
            label=f"Trajectory using {self.model_2_name}"
        )
        
        # Pad the shorter action log with zeros so both action plots span the same time axis.
        T1 = u_1.shape[1]
        T2 = u_2.shape[1]
        T_max = max(T1, T2)

        # Pad u_1 or u_2 if necessary
        if T1 < T_max:
            pad_width = ((0, 0), (0, T_max - T1))  # pad along time axis
            u_1 = np.pad(u_1, pad_width, mode='constant', constant_values=0)
        if T2 < T_max:
            pad_width = ((0, 0), (0, T_max - T2))
            u_2 = np.pad(u_2, pad_width, mode='constant', constant_values=0)

        # Plotting action plots
        self.ax_acc.plot(np.arange(u_1.shape[1]), u_1[0, :], color='forestgreen', label=f"{self.model_1_name}")
        self.ax_acc.plot(np.arange(u_2.shape[1]), u_2[0, :], color='crimson', label=f"{self.model_2_name}")

        self.ax_df.plot(np.arange(u_1.shape[1]), u_1[1, :], color='forestgreen', label=f"{self.model_1_name}")
        self.ax_df.plot(np.arange(u_2.shape[1]), u_2[1, :], color='crimson', label=f"{self.model_2_name}")
        
        self.ax_dr.plot(np.arange(u_1.shape[1]), u_1[2, :], color='forestgreen', label=f"{self.model_1_name}")
        self.ax_dr.plot(np.arange(u_2.shape[1]), u_2[2, :], color='crimson', label=f"{self.model_2_name}")
        self.setup_axes()

    def save_plot_and_close(self, filename, dpi=300, bbox_inches='tight'):
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(self.fig)
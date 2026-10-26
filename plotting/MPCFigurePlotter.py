import matplotlib.pyplot as plt
import numpy as np


class MPCFigurePlotter:
    """Thesis figure: one run on one track (path on top, the three MPC actions below)."""
    def __init__(self, track, model_name, track_name):
        # Initialize figure and create a grid with 2 rows, 3 columns
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 3)

        self.ax_track = self.fig.add_subplot(gs[0, :])
        self.ax_acc = self.fig.add_subplot(gs[1, 0])
        self.ax_df = self.fig.add_subplot(gs[1, 1])
        self.ax_dr = self.fig.add_subplot(gs[1, 2])
        self.model_name = model_name
        self.track_name = track_name
        self.track = track

    def setup_axes(self):
        self.ax_track.axis("equal")
        self.ax_track.set_xlabel("x (m)")
        self.ax_track.set_ylabel("y (m)")
        self.ax_track.set_title(f"{self.model_name} on Track {self.track_name}")
        self.ax_track.legend()

        self.ax_acc.set_xlabel("Timestep (0.1s)")
        self.ax_acc.set_ylabel("Acceleration ($m/s^2$)")

        self.ax_df.set_ylabel("Front Steering (radians)")
        self.ax_df.set_xlabel("Timestep (0.1s)")

        self.ax_dr.set_xlabel("Timestep (0.1s)")
        self.ax_dr.set_ylabel("Rear Steering (radians)")

    def plot_track(self, track):
        self.ax_track.plot(track[0, :], track[1, :], "b", label="Track")

    def plot_run(self, x_sim, u_mpc):
        """
        Params:
            x_sim: (4, n + 1) states [x, y, v, yaw]
            u_mpc: (3, n) actions (acceleration, front steering, rear steering)
        """
        self.plot_track(self.track)
        self.ax_track.scatter(x_sim[0], x_sim[1], s=0.5, color='red', zorder=0.5, label="Robot trajectory per timestep")
        self.ax_track.plot(x_sim[0], x_sim[1], color='green', zorder=-1, label="Robot trajectory")

        steps = np.arange(u_mpc.shape[1])
        self.ax_acc.plot(steps, u_mpc[0])
        self.ax_df.plot(steps, u_mpc[1])
        self.ax_dr.plot(steps, u_mpc[2])

        self.setup_axes()

    def save_plot_and_close(self, filename, dpi=300, bbox_inches='tight'):
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(self.fig)

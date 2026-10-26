import matplotlib.pyplot as plt
import numpy as np

import imageio
from natsort import natsorted
import glob
import os
import cv2


class MPCFigurePlotter:
    def __init__(self, track, sim_duration, model_name, track_name):
        # Initialize figure and create a grid with 2 rows, 3 columns
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 3)

        self.ax_track = self.fig.add_subplot(gs[0, :])
        self.ax_acc = self.fig.add_subplot(gs[1, 0])
        self.ax_df = self.fig.add_subplot(gs[1, 1])        
        self.ax_dr = self.fig.add_subplot(gs[1, 2])
        self.model_name = model_name
        self.track_name = track_name

        if len(track) != 0:
            self.track = track
            self.plot_track(track)

        # For storing incoming data
        self.X = np.zeros((4, sim_duration))
        self.U = np.zeros((3, sim_duration))
        
        self.current_timestep = 0
    
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

    def clear_axes(self):
        self.ax_track.clear()
        self.ax_acc.clear()
        self.ax_df.clear()
        self.ax_dr.clear()

    def plot_track(self, track):
        self.ax_track.plot(track[0, :], track[1, :], "b", label="Track")

    def plot_new_data(self, c_state, actions, i_timestep):
        """
        Params:
            c_state: cartesian state (x, y) of where robot currently is
            actions: only accepts actions of following form (acceleration, front steering, rear steering)
        """
        # Updates incoming data
        self.X[:, i_timestep] = c_state
        self.U[:, i_timestep] = actions
        
        N_timesteps = np.arange(i_timestep)

        self.clear_axes()

        self.plot_track(self.track)
        self.ax_track.scatter(self.X[0, :i_timestep], self.X[1, :i_timestep], s=0.5, color='red', zorder=0.5, label="Robot trajectory per timestep")
        self.ax_track.plot(self.X[0, :i_timestep], self.X[1, :i_timestep], color='green', zorder=-1, label="Robot trajectory")
        
        self.ax_acc.plot(N_timesteps, self.U[0, :i_timestep])
        self.ax_df.plot(N_timesteps, self.U[1, :i_timestep])
        self.ax_dr.plot(N_timesteps, self.U[2, :i_timestep])

        self.setup_axes()
        
        # plt.savefig(f"results/frames/frame_{i_timestep}.png")

    def convert_frames_into_video(self, fp_video):
        # Read folder with all the frames.
        frame_dir = "results/frames"
        frame_files = natsorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
        
        # Read first frame to get size
        first_frame = cv2.imread(frame_files[0])
        height, width, _ = first_frame.shape
        out = cv2.VideoWriter(fp_video, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))

        # Write each frame to the video
        for frame_file in frame_files:
            img = cv2.imread(frame_file)
            out.write(img)
            os.remove(frame_file)

        out.release()

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

    def save_plot_and_close(self, filename, dpi=300, bbox_inches='tight'):
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(self.fig)
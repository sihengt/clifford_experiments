import matplotlib.pyplot as plt
import numpy as np

import imageio
from natsort import natsorted
import glob
import os
import cv2


class MPCFigureComparisonPlotter:
    def __init__(self, track, sim_duration, model_1_name, model_2_name, track_name):
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

        if len(track) != 0:
            self.track = track
            self.plot_track(track)

        # For storing incoming data
        self.X_1 = np.zeros((4, sim_duration))
        self.X_2 = np.zeros((4, sim_duration))
        self.U_1 = np.zeros((3, sim_duration))
        self.U_2 = np.zeros((3, sim_duration))

        self.current_timestep = 0
    
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
        # Updates incoming data

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
        
        # How do I check self.U_1.shape[1] or self.U_2.shape, which one is longer, then pad zeros on the shorter one so they're the same length?
        # Find max length
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
        
        # plt.savefig(f"results/frames/frame_{i_timestep}.png")

    def plot_new_data(self, x_k_1, x_k_2, u_k_1, u_k_2, i_timestep):
        """
        Params:
            c_state: cartesian state (x, y) of where robot currently is
            actions: only accepts actions of following form (acceleration, front steering, rear steering)
        """
        # Updates incoming data
        self.X_1[:, i_timestep] = x_k_1
        self.X_2[:, i_timestep] = x_k_2
        self.U_1[:, i_timestep] = u_k_1
        self.U_1[:, i_timestep] = u_k_2
        
        N_timesteps = np.arange(i_timestep)

        self.clear_axes()

        self.plot_track(self.track)

        # Plotting robot 1's trajectory
        self.ax_track.scatter(
            self.X_1[0, :i_timestep],
            self.X_1[1, :i_timestep],
            s=0.5,
            color='crimson',
            zorder=0.5,
            label=f"Trajectory per timestep using {self.model_1_name}"
        )
        self.ax_track.plot(
            self.X_1[0, :i_timestep],
            self.X_1[1, :i_timestep],
            color='forestgreen',
            zorder=-1,
            label=f"Trajectory using {self.model_1_name}"
        )

        # Plotting robot 2's trajectory
        self.ax_track.scatter(
            self.X_2[0, :i_timestep],
            self.X_2[1, :i_timestep],
            s=0.5,
            color='navy',
            zorder=0.5,
            label=f"Trajectory per timestep using {self.model_2_name}"
        )
        
        self.ax_track.plot(
            self.X_2[0, :i_timestep],
            self.X_2[1, :i_timestep],
            color='goldenrod',
            zorder=-1,
            label=f"Trajectory using {self.model_2_name}"
        )
        
        # Plotting action plots
        self.ax_acc.plot(N_timesteps, self.U_1[0, :i_timestep], color='', label=f"{self.model_1_name}")
        self.ax_acc.plot(N_timesteps, self.U_2[0, :i_timestep], color='', label=f"{self.model_2_name}")

        self.ax_df.plot(N_timesteps, self.U_1[1, :i_timestep], color='', label=f"{self.model_1_name}")
        self.ax_df.plot(N_timesteps, self.U_2[1, :i_timestep], color='', label=f"{self.model_2_name}")
        
        self.ax_dr.plot(N_timesteps, self.U_1[2, :i_timestep], color='', label=f"{self.model_1_name}")
        self.ax_dr.plot(N_timesteps, self.U_2[2, :i_timestep], color='', label=f"{self.model_2_name}")
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
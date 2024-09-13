import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from utils.plot_car import get_arrow_marker_car

class CarPlot:
    def __init__(self, wheel_base, title="2D Car Pose"):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("Position (x)")
        self.ax.set_ylabel("Position (y)")
        self.ax.legend()
        self.ax.axis('equal')

        self.car_marker = None
        self.car_goal_marker = None

        self.car_goal_marker_color = 'r'

        # Car trajectory properties
        self.car_ref_traj = None
        self.car_ref_traj_color = 'green'  
        self.car_ref_traj_linewidth = 2 
        self.car_ref_traj_alpha = 0.8

        # (Moving) Car marker properties
        self.width = 0.3
        self.head_width = 0.3
        self.head_length = 0.3
        self.car_marker_color = 'b'
        self.car_wheel_base = wheel_base

    def plot_ref_traj(self, traj):
        if self.car_ref_traj:
            self.car_ref_traj.remove()
        self.car_ref_traj, = self.ax.plot(
            traj[:, 0],
            traj[:, 1],
            self.car_ref_traj_color,
            label="Reference Trajectory",
            linewidth=self.car_ref_traj_linewidth,
            alpha=self.car_ref_traj_alpha)

    def update_car_marker(self, car_marker_name, car_marker_props):
        car_marker = getattr(self, car_marker_name)
        if car_marker:
            car_marker.remove()
        arrow = FancyArrow( car_marker_props['x'],
                                    car_marker_props['y'],
                                    car_marker_props['dx'],
                                    car_marker_props['dy'],
                                        width=self.width,
                                        head_width=self.head_width,
                                        head_length=self.head_length,
                                        fc=car_marker_props['car_marker_color'],
                                        ec=car_marker_props['car_marker_color'])
        self.ax.add_patch(arrow)
        setattr(self, car_marker_name, arrow)

    def plot_car(self, state, is_goal=False):
        x, y, theta = state
        marker_props = {}
        marker_props['x'], marker_props['y'], marker_props['dx'], marker_props['dy'] \
            = get_arrow_marker_car(x, y, theta, self.car_wheel_base)
        if is_goal:
            marker_props['car_marker_color'] = self.car_goal_marker_color
            self.update_car_marker("car_marker", marker_props)
        else:
            marker_props['car_marker_color'] = self.car_marker_color
            self.update_car_marker("car_goal_marker", marker_props)

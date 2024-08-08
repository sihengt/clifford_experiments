import numpy as np
import matplotlib.pyplot as plt

def get_arrow_marker_car(x, y, theta, l):
    arrow_base_x = x - l/2 * np.cos(theta)
    arrow_base_y = y - l/2 * np.sin(theta)
    dx = l * np.cos(theta)
    dy = l * np.sin(theta)
    
    return arrow_base_x, arrow_base_y, dx, dy


def plot_car(start, target, wheel_base, cen_x, cen_y, turn_radius, show_plot=False):
    plt.clf()
    car_x, car_y, car_dx, car_dy= get_arrow_marker_car(start[0], start[1], start[2], wheel_base/2)
    target_x, target_y, target_dx, target_dy = get_arrow_marker_car(target[0], target[1], target[2], wheel_base/2)
    plt.arrow(car_x, car_y, car_dx, car_dy, width=0.3, head_width=0.3, head_length=0.3, fc='green', ec='g')
    plt.arrow(target_x, target_y, target_dx, target_dy, width=0.3, head_width=0.3, head_length=0.3, fc='red', ec='red')
    plt.plot([car_x + car_dx / 2, target_x + target_dx / 2], [car_y + car_dy / 2, target_y + target_dy / 2], 'b')
    
    circle = plt.Circle((cen_x, cen_y), turn_radius, edgecolor='b', facecolor='none')
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')
    plt.grid(True)

    if show_plot:
        plt.show()
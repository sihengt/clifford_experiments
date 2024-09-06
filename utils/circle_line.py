import torch
import numpy as np
import matplotlib.pyplot as plt

def draw_circle(x, y, r, circle_color):
    xs = []
    ys = []
    # That little extra to close the circle
    angles = np.arange(0, 2.02*np.pi, 0.02)
    for angle in angles:
        xs.append(r * np.cos(angle) + x)
        ys.append(r * np.sin(angle) + y)
    plt.plot(xs, ys, '-', color=circle_color)

def find_circle_line_intersection(p1, p2, c_center, radius):
    """
    Calculates where a circle and line intersect, bounded by both points. Intended to be used with Pure Pursuit, where 
    the algorithm can find the closest point of the trajectory to follow.
    """
    p1_pos = p1[:2]
    p2_pos = p2[:2]

    # Moving points to the center of the circle 
    p1_offset = p1_pos - c_center
    p2_offset = p2_pos - c_center
    x1, y1 = p1_offset
    x2, y2 = p2_offset
    
    d_x = x2 - x1
    d_y = y2 - y1
    d_r = torch.sqrt(d_x**2 + d_y**2)
    
    # determinant
    D = x1 * y2 - x2 * y1
    discriminant = radius**2 * d_r**2 - D**2
    
    # No intersection
    if (discriminant < 1e-9):
        return []
    
    else:
        x_int_1 = (D * d_y + torch.sign(d_y) * d_x * torch.sqrt(discriminant)) / d_r ** 2
        x_int_2 = (D * d_y - torch.sign(d_y) * d_x * torch.sqrt(discriminant)) / d_r ** 2
        y_int_1 = (-D * d_x + torch.abs(d_y) * torch.sqrt(discriminant)) / d_r ** 2
        y_int_2 = (-D * d_x - torch.abs(d_y) * torch.sqrt(discriminant)) / d_r ** 2

        return [
            torch.stack([x_int_1, y_int_1]) + c_center,
            torch.stack([x_int_2, y_int_2]) + c_center
        ]    

if __name__ == "__main__":
    print("Hello")
    point_1         = np.array([2, 3])
    point_2         = np.array([-2, -3])
    circle_center   = np.array([-6, -9])
    circle_radius   = 1
    intersections = find_circle_line_intersection(point_1, point_2, circle_center, 1)
    print("{} intersections found.".format(len(intersections)))

    draw_circle(circle_center[0], circle_center[1], circle_radius, 'r')
    plt.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]])
    for point in intersections:
        plt.scatter(point[0], point[1])
        
    plt.axis('equal')
    plt.show()
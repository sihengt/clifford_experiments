import numpy as np
import matplotlib.pyplot as plt
import math

def add_line (path) : 
    for i in range (0,len(path)):
        plt.plot(path[i][0],path[i][1],'.',color='red',markersize=10)
    
    for i in range(0,len(path)-1):
        plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],color='b')
        
        plt.axis('scaled')
        # plt.show()

def add_complicated_line (path,lineStyle,lineColor,lineLabel) :
    for i in range (0,len(path)):
        plt.plot(path[i][0],path[i][1],'.',color='red',markersize=10)
        
    for i in range(0,len(path)-1):
        if(i == 0):
            # plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],color='b')
            plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],lineStyle,color=lineColor,label=lineLabel)    
        else:
            plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],lineStyle,color=lineColor)        
            
    plt.axis('scaled')
            
def highlight_points (points, pointColor):
    for point in points :
        plt.plot(point[0], point[1], '.', color = pointColor, markersize = 10)
        
def draw_circle (x, y, r, circleColor):
    xs = []
    ys = []
    angles = np.arange(0, 2.2*np.pi, 0.2)        
    
    for angle in angles :
        xs.append(r*np.cos(angle) + x)
        ys.append(r*np.sin(angle) + y)
        
    plt.plot(xs, ys, '-', color = circleColor)

# helper function: sgn(num)
# returns -1 if num is negative, 1 otherwise
def sgn (num):
    if num >= 0:
        return 1
    else:
        return -1
    
# currentPos: [currentX, currentY]
# pt1: [x1, y1]
# pt2: [x2, y2]
def line_circle_intersection (currentPos, pt1, pt2, lookAheadDis):

    # extract currentX, currentY, x1, x2, y1, and y2 from input arrays
    currentX = currentPos[0]
    currentY = currentPos[1]
    x1 = pt1[0]  
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]  
    
    # boolean variable to keep track of if intersections are found
    intersectFound = False  
    
    # output (intersections found) should be stored in arrays sol1 and sol2  
    # if two solutions are the same, store the same values in both sol1 and sol2  
    
    # subtract currentX and currentY from [x1, y1] and [x2, y2] to offset the system to origin  
    x1_offset = x1 - currentX  
    y1_offset = y1 - currentY  
    x2_offset = x2 - currentX  
    y2_offset = y2 - currentY  
    
    # calculate the discriminant using equations from the wolframalpha article
    dx = x2_offset - x1_offset
    dy = y2_offset - y1_offset
    dr = math.sqrt (dx**2 + dy**2)
    D = x1_offset*y2_offset - x2_offset*y1_offset
    discriminant = (lookAheadDis**2) * (dr**2) - D**2  
    
    # if discriminant is >= 0, there exist solutions
    if discriminant >= 0:
        intersectFound = True
    
        # calculate the solutions
        sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
        sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
        sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
        sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2    
    
        # add currentX and currentY back to the solutions, offset the system back to its original position
        sol1 = [sol_x1 + currentX, sol_y1 + currentY]
        sol2 = [sol_x2 + currentX, sol_y2 + currentY]  
    
    # graphing functions to visualize the outcome  
    # ---------------------------------------------------------------------------------------------------------------------------------------  
    plt.plot ([x1, x2], [y1, y2], '--', color='grey')
    draw_circle (currentX, currentY, lookAheadDis, 'orange')
    if intersectFound == False :
        print ('No intersection Found!')
    else:
        print ('Solution 1 found at [{}, {}]'.format(sol1[0], sol1[1]))
        print ('Solution 2 found at [{}, {}]'.format(sol2[0], sol2[1]))
        plt.plot (sol1[0], sol1[1], '.', markersize=10, color='red', label='sol1')
        plt.plot (sol2[0], sol2[1], '.', markersize=10, color='blue', label='sol2')
        plt.legend()
        
    plt.axis('scaled')
    plt.show()
        
# now call this function and see the results!
line_circle_intersection ([0, 1], [2, 3], [-2, -4], 1)
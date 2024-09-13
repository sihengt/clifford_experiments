from .CarPlot import CarPlot
import matplotlib.pyplot as plt

class PurePursuitPlot(CarPlot):
    def __init__(self, wheel_base, lookahead_distance):
        super().__init__(wheel_base)
        
        self.purePursuitCircle = None
        self.purePursuitCircleColor = 'orange'
        self.lookaheadPoint = None

        self.lookaheadDistance = lookahead_distance
    
    def plot_pure_pursuit(self, state, ref_state):
        # Drawing / re-drawing the circle
        if self.purePursuitCircle:
            self.purePursuitCircle.remove()
        self.purePursuitCircle = plt.Circle(
            state,
            self.lookaheadDistance,
            edgecolor=self.purePursuitCircleColor,
            facecolor='none'
        )
        self.ax.add_patch(self.purePursuitCircle)

        # Drawing lookahead point
        if self.lookaheadPoint:
            self.lookaheadPoint.remove()
        self.lookaheadPoint = self.ax.scatter(
            ref_state[0],
            ref_state[1],
            color=self.purePursuitCircleColor,
            s=5,
            label="Lookahead point")
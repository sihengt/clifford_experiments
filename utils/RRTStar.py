import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.purePursuit import purePursuit, wrapAng
import utils.doubleSteerKinematics as dsk
import matplotlib.pyplot as plt
from datetime import datetime
from heapq import *
import pickle

def dummyCollisionCheck(trajRef, predInit=None):
    return True, None, None 

def pickleFile(data, base_filename, filepath):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{base_filename}"
    fp = os.path.join(filepath, filename)
    with open(fp, 'wb') as f:
        pickle.dump(data, f)

class Node:
    def __init__(self, state, parent=None, pathFromParent=None, pathDist=None, predState=None):
        self.state = state
        self.parent = None
        self.cost = 0
        self.children = set()
        self.pathFromParent = None
        self.predState=predState
        if not parent is None:
            self.addParent(parent, pathFromParent, pathDist)
        if self.pathFromParent is None:
            self.pathFromParent = torch.tensor([])

    def addParent(self,parent,pathFromParent,pathDist):
        self.parent = parent
        self.parent.children.add(self)
        self.pathFromParent = pathFromParent
        self.changeCost(parent.cost + pathDist)

    def replaceParent(self, newParent, pathFromParent, pathDist):
        self.parent.children.remove(self)
        self.parent = newParent
        self.pathFromParent = pathFromParent
        self.parent.children.add(self)
        self.changeCost(newParent.cost + pathDist)

    def changeCost(self,newCost):
        costOffset = newCost - self.cost
        self.cost = newCost
        for child in self.children:
            child.changeCost(child.cost+costOffset)

class RRTStar:
    def __init__(self, pp, speedRange, steerDist, rewireDist, p_goal, goalTol, thresh=0.5, lr=0.01, epochs=150, path_weight=10, div_weight=1, numCollisionAttempts=torch.inf, steerLimit=1, maxIterations=torch.inf, collisionCheck=None, debugFP=""):
        self.pp = pp
        self.speedRange = speedRange
        self.steerDist = steerDist
        self.rewireDist = rewireDist
        self.p_goal = p_goal
        self.maxIterations = maxIterations
        self.collisionCheck = collisionCheck
        self.goalTol = goalTol
        self.numCollisionAttempts=numCollisionAttempts
        self.steerLimit=steerLimit
        self.thresh = thresh
        self.lr = lr
        self.epochs = epochs
        self.path_weight = path_weight
        self.div_weight = div_weight
        self.debugFP = debugFP
        if self.collisionCheck is None:
            self.collisionCheck = dummyCollisionCheck

    def plotSearch(self,xBounds,yBounds):
        plt.clf()
        plt.arrow(self.start[0],self.start[1],self.start[2].cos(),self.start[2].sin(),head_width=0.1,head_length=0.1,fc='green',ec='green')
        goalCircle = plt.Circle(self.goal[:2],self.goalTol, edgecolor = 'none', facecolor='red')
        plt.gca().add_patch(goalCircle)
        #plt.arrow(self.goal[0],self.goal[1],self.goal[2].cos(),self.goal[2].sin(),head_width=0.1,head_length=0.1,fc='red',ec='red')
        plt.axis([xBounds[0],xBounds[1],yBounds[0],yBounds[1]])
        for node in self.nodes:
            state = node.state
            plt.arrow(state[0],state[1],state[2].cos()/2.0,state[2].sin()/2.0,head_width=0.1,head_length=0.1,fc='black',ec='black')
            path = node.pathFromParent
            if len(path.shape) > 1:
                plt.plot(path[:,0],path[:,1],'black',linewidth=1)

        plt.pause(0.01)
    
    def backward_node(self, node):
        if node.parent == None:
            return [node]

        path = self.backward_node(node.parent)
        path.append(node)

        return path

    def backward(self,node):
        if len(node.pathFromParent.shape) < 2:
            return [torch.tensor([])]

        path = self.backward(node.parent)
        path.append(node.pathFromParent)

        return path

    def search(self, start, goal, xBounds, yBounds, plot=False):
        self.start = start
        self.goal = goal
        self.nodes = [Node(self.start)]
        self.foundGoal = False
        while len(self.nodes) < self.maxIterations:
            # What is predState: 
            if len(self.nodes) > self.maxIterations * 0.9:
                self.p_goal = 0.9

            path, pathDist, parentIndex, predState = self.steerToRandState_collision(xBounds, yBounds)
            parentNode = self.nodes[parentIndex]

            newState = path[-1]
            newNode = Node(newState, parentNode, path, pathDist, predState)
            
            # rewire tree if necessary
            # self.rewire(newNode)
            self.nodes.append(newNode)
            if torch.norm(newState[:2] - self.goal[:2]) < self.goalTol:
                break
            
            if plot:
                print(f"\rNodes in RRT: {len(self.nodes)}", end="")
                self.plotSearch(xBounds,yBounds)

        # Assume node closest to goal is the solution
        min_dist = torch.inf
        for node in self.nodes:
            dist = torch.norm(node.state[:2] - self.goal[:2])
            if dist < min_dist:
                min_dist = dist
                min_node = node
        path = self.backward(min_node)
        path = torch.cat(path, dim=0)
        
        rrt_nodes = self.backward_node(min_node)
        pruned_path = self.prune(rrt_nodes)
        pruned_path = torch.cat(pruned_path, dim=0)

        # (For debugging) Saves RRT solutions into a pickle if a filepath is specified.
        if len(self.debugFP) > 0:
            print("Pickling to {}".format(self.debugFP))
            pickleFile(rrt_nodes, "rrt.pkl", self.debugFP)
        
        # optimized_path = self.optimize(pruned_path)

        # return path, pruned_path, optimized_path
        return path

    def prune(self, sol_nodes):
        current_index = 0
        
        # These variables are only updated when a shorter path is found.
        next_index = -1
        current_path = None
        current_predState = None
        
        while current_index < len(sol_nodes):
            # Check for the farthest reachable node
            for i in range(current_index + 2, len(sol_nodes)):
                # "Parent" is the state you are steering from
                parent = sol_nodes[current_index]
                # "Candidate" is the state you are trying to steer to
                candidate = sol_nodes[i]    

                # Path from parent, if this path is accepted.
                nodePath, nodeDist = self.pp.calcSteerTraj(parent.state, candidate.state, 0.05, forward_only=True)

                if nodePath is None:
                    continue

                # Checking for divergence constraint, accounting for first node.
                with torch.no_grad():
                    if parent.predState is None:
                        divConstraintSatisfied, _, predState= self.collisionCheck(nodePath)
                    else:
                        divConstraintSatisfied, _, predState= self.collisionCheck(nodePath, parent.predState)
                
                if divConstraintSatisfied:
                    current_path = nodePath
                    current_predState = predState
                    next_index = i

            # Updating node parents.
            if not current_path is None and not current_predState is None and next_index != -1:
                sol_nodes[next_index].parent = sol_nodes[current_index]
                sol_nodes[next_index].pathFromParent = current_path
                sol_nodes[next_index].predState = current_predState
                current_index = next_index

                # Resetting.
                current_path = None
                next_index = -1

            else:
                # We need to check the next index manually if next_index wasn't updated.
                current_index += 1
            
        try:
            prunedTraj = self.backward(sol_nodes[-1])
        except RecursionError:
            breakpoint()

        return prunedTraj

    """
    Optimizes path generated by RRT using a distance cost and a relaxed log barrier function defined using divergence 
    metric.

    Inputs:
        path: a tensor of shape (N, 3) corresponding to the (x, y, yaw)
    """
    def optimize(self, path):
        # Optimizing over all states between first and last state to keep start / end points untouched.
        path_first_state  = path[0]
        path_last_state   = path[-1]
        path_intermediate = path[1:-1].clone()
        path_intermediate.requires_grad = True
        
        optimizer = torch.optim.Adam([path_intermediate], lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Recreate the full path with updated intermediate points
            full_path = torch.cat((path_first_state.unsqueeze(0), path_intermediate, path_last_state.unsqueeze(0)), dim=0)
            
            # Get path length cost to minimize path distance. When used alone, generates a straight line between start / end.
            # First term computes path[i] - path[i-1], second term computes path[i+1] - path[i]
            diff = torch.sum(torch.diff(full_path[:, :2], dim=0)[:-1, :] ** 2, dim=1) + torch.sum(torch.diff(full_path[1:, :2], dim=0) ** 2, dim=1)
            path_length = torch.sum(diff)

            # Get divergence for constraint.
            _, _, _, max_divergence = self.collisionCheck(full_path, returnDivergence=True)
            if max_divergence > self.thresh:
                div_loss = -torch.log(max_divergence)
            else:
                div_loss = torch.exp(1 - (max_divergence / self.thresh))

            # Loss = weighted sum of the costs
            loss = self.path_weight * path_length + self.div_weight * div_loss
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item()}')
        
        return torch.cat((path_first_state.unsqueeze(0), path_intermediate, path_last_state.unsqueeze(0)), dim=0)
            
    def rewire(self, newNode):
        speed = 0.5

        # Gets distance and path from new node to all pre-existing nodes
        TreeDist, TreePath = self.TreePurePursuit(newNode.state, speed)
        
        for dist, path, node in zip(TreeDist, TreePath, self.nodes):
            if dist > self.rewireDist or node.cost <= newNode.cost + dist:
                continue
            
            # collisionPath = torch.cat((newNode.pathFromParent, path),dim=0)
            
            with torch.no_grad():
                divConstraintSatisfied, _, predState = self.collisionCheck(path, node.predState)
            
            if not divConstraintSatisfied:
                continue

            node.replaceParent(newNode,path,dist)

    def steerToRandState_collision(self, xBounds, yBounds):
        """
        Samples a random state within x/yBounds. Calculates the distance and a path (through PurePursuit) 
        to all current nodes. Sorts distances and corresponding paths according to minimum distance to 
        random state. Check if divergence constraint is satisfied. Return first path where divergence is satisfied.

        Returns:
            path: path with number of steps 
            speed * numSteps: total distance covered
            minIndex: Index of node closest to sampled point.
            predState: predState from LSTM.
        """
        path = None
        while path is None or path.shape[0] == 0:
            # With small probability sample directly toward the goal
            if not self.foundGoal and torch.rand(1) < self.p_goal:
                random_state = self.sample_goal()
            # otherwise, sample a point within bounds to build tree towards.
            else:
                random_state = self.sample(xBounds,yBounds)
            
            # numSteps is used to constrain length of each path
            speed = torch.rand(1)*(self.speedRange[1] - self.speedRange[0]) + self.speedRange[0]
            numSteps = int(torch.floor(self.steerDist / speed))
            
            # TreeDist/Path = dist/path from all self.nodes to sampled state.
            # TODO: can we rethink how we build the tree?
            TreeDist, TreePath = self.TreePurePursuit(random_state, speed)
            indexedDists = [[TreeDist[i], i] for i in range(len(TreeDist))]
            heapify(indexedDists)
            
            # Samples a random state and finds nodes closest to the current state.
            for _ in range(min(self.numCollisionAttempts, len(indexedDists))):
                # minIndex contains the parent node index (lowest distance to sampled node)
                _, minIndex = heappop(indexedDists)

                # distance of inf indicates invalidity
                if not TreeDist[minIndex] < torch.inf:
                    break
                
                # Checking the candidate path (extended by numSteps) from the candidate parent
                candidateParent = self.nodes[minIndex]
                candidatePath = TreePath[minIndex][:numSteps]

                with torch.no_grad():
                    divConstraintSatisfied, _, predState = self.collisionCheck(candidatePath, candidateParent.predState)
                if divConstraintSatisfied:
                    path = candidatePath
                    break
        
        return path, speed * numSteps, minIndex, predState
    
    def steerToRandState(self,xBounds,yBounds):
        path = None
        while path is None or path.shape[0] == 0:
            if not self.foundGoal and torch.rand(1) < self.p_goal:
                randState = self.sample_goal()
            else:
                randState =  self.sample(xBounds,yBounds)
            speed = torch.rand(1)*(self.speedRange[1]-self.speedRange[0]) + self.speedRange[0]
            #speed = torch.tensor(0.1)
            TreeDist,TreePath = self.TreePurePursuit(randState,speed)
            minIndex = np.argmin(TreeDist)
            if TreeDist[minIndex] < torch.inf:
                path = TreePath[minIndex]

        numSteps = int(torch.floor(self.steerDist/speed))
        path = path[:numSteps,:]
        return path,TreeDist[minIndex],minIndex

    """
    For a given state and speed, iterates through all pre-existing states and finds a  
    path to each state.

    Returns:
        dists: Distances from each pre-exiting node to state.
        paths: The path from each pre-existing node to state (if treeToState)
    """
    def TreePurePursuit(self, state, speed=0.1, treeToState=True, forward_only=True):
        dists = []
        paths = []
        # Iterating through pre-existing states (self.nodes) and getting distance to each node.
        for node in self.nodes:
            if treeToState:
                nodePath, nodeDist = self.pp.calcSteerTraj(node.state, state, speed, forward_only=forward_only)
            else:
                nodePath, nodeDist = self.pp.calcSteerTraj(state, node.state, speed, forward_only=forward_only)
            dists.append(nodeDist)
            paths.append(nodePath)
        return dists, paths

    def sample_goal(self):
        """ Sets goal as the sample """
        sample = self.goal
        return sample
    
    def sample(self,xBounds,yBounds):
        xSample = torch.rand(1)*(xBounds[1]-xBounds[0])+xBounds[0]
        ySample = torch.rand(1)*(yBounds[1]-yBounds[0])+yBounds[0]
        thetaSample = torch.rand(1)*2*torch.pi-torch.pi

        # convert to discrete
        sample = torch.cat((xSample,ySample,thetaSample),dim=-1)
        return sample

if __name__ == "__main__":
    pp_params = {'wheelBase': 0.75,
                'steerScale': 0.75,
                'throttleScale': 1}
    pp = purePursuit(**pp_params)
    rrt_params = {'speedRange':[0.05,0.1],
                'steerDist': 2.5,
                'rewireDist': 5,
                'p_goal': 0.1,
                'stateWeight': [1,1,5],
                'samplingRes': [0.1,0.1,0.1],
                'maxIterations': 100,
                }

    xBounds = torch.tensor([-1,10])
    yBounds = torch.tensor([-5,5])

    rrt = RRTStar(pp,**rrt_params)

    for i in range(100):
        start = rrt.sample(xBounds,yBounds)
        target = rrt.sample(xBounds,yBounds)
        result = rrt.search(start,target,xBounds,yBounds)
        rrt.plotSearch(xBounds,yBounds)
        plt.plot(result[:,0],result[:,1],'green',linewidth=5)
        plt.pause(1)



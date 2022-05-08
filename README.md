# Breadth First Search
## AIM

To develop an algorithm to find the route from the source to the destination point using breadth-first search.

## THEORY
Breadth-first search, also known as BFS, finds shortest paths from a given source vertex to all other vertices, in terms of the number of edges in the paths.

## DESIGN STEPS

### STEP 1:
Identify a location in the google map:

### STEP 2:
Select a specific number of nodes with distance

### STEP 3:
Import required packages.

### STEP 4:
Include each node and its distance separately in the dictionary data structure.

### STEP 5:
End of program.


## ROUTE MAP
#### Example map
![Screenshot (286)](https://user-images.githubusercontent.com/75236145/166117310-3cf0ff9c-1f7d-4278-a344-3e454584b5f0.png)


## PROGRAM
```python
Student name : M VIGNESH
Reg.no : 212220233002
 ```
 ```python
 %matplotlib inline
import matplotlib.pyplot as plt
import random
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations

class Problem(object):
    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        
        raise NotImplementedError
    def result(self, state, action): 
        raise NotImplementedError
    def is_goal(self, state):        
        return state == self.goal
    def action_cost(self, s, a, s1): 
        return 1
    
    def __str__(self):
        return '{0}({1}, {2})'.format(
            type(self).__name__, self.initial, self.goal)
            
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __str__(self): 
        return '<{0}>'.format(self.state)
    def __len__(self): 
        return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): 
        return self.path_cost < other.path_cost
        
failure = Node('failure', path_cost=math.inf) 
cutoff  = Node('cutoff',  path_cost=math.inf)

def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)
        

def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []  
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state]
    
FIFOQueue = deque

def breadth_first_search(problem):
    "Search shallowest nodes in the search tree first."
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node
    frontier = FIFOQueue([node])
    reached = {problem.initial}
    while frontier:
        node = frontier.pop()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
    return failure
    
class RouteProblem(Problem):
    """A problem to find a route between locations on a `Map`.
    Create a problem with RouteProblem(start, goal, map=Map(...)}).
    States are the vertexes in the Map graph; actions are destination states."""
    
    def actions(self, state): 
        """The places neighboring `state`."""
        return self.map.neighbors[state]
    
    def result(self, state, action):
        """Go to the `action` place, if the map says that is possible."""
        return action if action in self.map.neighbors[state] else state
    
    def action_cost(self, s, action, s1):
        """The distance (cost) to go from s to s1."""
        return self.map.distances[s, s1]
    
    def h(self, node):
        "Straight-line distance between state and the goal."
        locs = self.map.locations
        return straight_line_distance(locs[node.state], locs[self.goal])
        
 class Map:
    """A map of places in a 2D world: a graph with vertexes and links between them. 
    In `Map(links, locations)`, `links` can be either [(v1, v2)...] pairs, 
    or a {(v1, v2): distance...} dict. Optional `locations` can be {v1: (x, y)} 
    If `directed=False` then for every (v1, v2) link, we add a (v2, v1) link."""

    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'): # Distances are 1 by default
            links = {link: 1 for link in links}
        if not directed:
            for (v1, v2) in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.neighbors = multimap(links)
        self.locations = locations or defaultdict(lambda: (0, 0))

        
def multimap(pairs) -> dict:
    "Given (key, val) pairs, make a dict of {key: [val,...]}."
    result = defaultdict(list)
    for key, val in pairs:
        result[key].append(val)
    return result
    
saveetha_nearby_locations = Map(
    {('SaveethaHospital', 'Chembarambakkam'):  7, ('Chembarambakkam', 'PoonamalleeBridge'): 6, ('PoonamalleeBridge', 'PoonamalleeBusTerminus'): 1, ('PoonamalleeBridge', 'SaveethaDentalCollege'): 3, ('PoonamalleeBusTerminus', 'Kattupakkam'): 2,
     ('PoonmalleeBusTerminal', 'Mangadu'):4, ('SaveethaDentalCollege', 'Kattupakkam'): 2, ('SaveethaDentalCollege', 'Maduravoyal'):  5, ('Maduravoyal', 'Koyambedu'): 5, ('Koyambedu', 'Vadapalani'): 5, ('Kattupakkam', 'Porur'): 4, 
     ('Porur', 'Vadapalani'): 8, ('Vadapalani', 'Guindy'):  8, ('Porur', 'Guindy'): 10, ('Mangadu', 'Kundrathur'): 4, ('Kundrathur', 'Porur'): 9, ('Kundrathur', 'Tiruneermalai'): 7, ('Kundrathur', 'Pammal'): 6, ('Pammal', 'Porur'): 10, ('Pammal', 'Guindy'): 14, ('Pammal', 'Airport'): 6, ('Tiruneermalai', 'Balaji college'): 2, ('Balaji college', 'Chrompet'): 2, ('Guindy', 'Velachery'): 4, ('Chrompet', 'Velachery'): 12})


r0 = RouteProblem('Maduravoyal', 'Velachery', map=saveetha_nearby_locations)
r1 = RouteProblem('Chembarambakkam', 'Guindy', map=saveetha_nearby_locations)
r2 = RouteProblem('PoonamalleeBridge', 'Kundrathur', map=saveetha_nearby_locations)
r3 = RouteProblem('SaveethaHospital', 'Kattupakkam', map=saveetha_nearby_locations)
r4 = RouteProblem('Mangadu', 'Guindy', map=saveetha_nearby_locations)

goal_state_path=breadth_first_search(r4)
path_states(goal_state_path) 
print("GoalStateWithPath:{0}".format(goal_state_path))
print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))

```

## OUTPUT:
![Screenshot (287)](https://user-images.githubusercontent.com/75236145/166117191-25038c57-7d49-4635-bbed-967d80def7cf.png)


## SOLUTION JUSTIFICATION:
Route follow the minimum distance between locations using breadth-first search.

## RESULT:
Thus an algorithm to find the route from the source to the destination point using breadth-first search is developed and executed successfully.

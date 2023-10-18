#thesis implementation

import matplotlib.pyplot as plt # for 2D visualization
from mpl_toolkits.mplot3d import Axes3D #for 3D visualization of drone trajectory
import numpy as np
import math
import heapq
import random
import time
import pandas as pd
import plotly.express as px
from djitellopy import Tello
import os
from scipy.spatial.transform import Rotation as R



def map_graph_generator(r,c,unitcost=1,blocked_nodes=[]):
  #r is number of rows of graph
  #c is the number of columns of graph
  #unitcost is the firect distance of 2 adjacent nodes
  #output is adjacensy matrix
  #(0,0) point is assumed as upper left point of grid.

  #adjacent nodes of target X:
  # 5 2 6
  # 1 X 3
  # 8 4 7

  matrix_dim = r*c #row and column
  adj_matrix = np.zeros([matrix_dim,matrix_dim]) + 9999999
  orthogonalcost = np.round(unitcost * math.sqrt(2))

  for i in range(0,matrix_dim-1):
    i_row_idx = i // c
    i_col_idx = i % c
    for j in range(i+1,matrix_dim):
      j_row_idx = j // c
      j_col_idx = j % c
      
      #case adjacent 1,2,3,4
      if((abs(i_row_idx - j_row_idx) + abs(i_col_idx - j_col_idx)) == 1):
        adj_matrix[i,j] = unitcost
        adj_matrix[j,i] = unitcost
      #case adjacent 5,6,7,8
      elif(abs(i_row_idx - j_row_idx) == 1) & (abs(i_col_idx - j_col_idx) == 1):
        adj_matrix[i,j] = orthogonalcost
        adj_matrix[j,i] = orthogonalcost
  #blocked
  for i in range(len(blocked_nodes)):
    adj_matrix[:,blocked_nodes[i]] = 9999999
    adj_matrix[blocked_nodes[i],:] = 9999999
  return adj_matrix

#A-Star:
def astar(graph, start, goal):
    frontier = [(0, start)]
    visited = set()
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        visited.add(current)

        for neighbor in range(len(graph[current])):
            if graph[current][neighbor] == float('inf'):
                continue
            new_cost = cost_so_far[current] + graph[current][neighbor]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal, graph)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    return None

def heuristic(a, b, graph):
    return graph[a,b]

#convert path to commands
def calcul_rotation(r,c,current_position,current_degree,destination):
  #this function generates counterclockwise rotation degree

  current_position_row_idx = current_position // c
  current_position_col_idx = current_position % c

  destination_row_idx = destination // c
  destination_col_idx = destination % c

  rotation_degree = 0

  #case adjacent 1,2,3,4
  if((abs(current_position_row_idx - destination_row_idx) + abs(current_position_col_idx - destination_col_idx)) == 1):
    #case 2
    if(current_position_row_idx - destination_row_idx == 1):
      rotation_degree = 90 - current_degree
      final_degree = 90
    #case 4
    elif(current_position_row_idx - destination_row_idx == -1):
      rotation_degree = 270 - current_degree
      final_degree = 270
    #case 1
    elif(current_position_col_idx - destination_col_idx == 1):
      rotation_degree = 180 - current_degree
      final_degree = 180
    #case 3
    elif(current_position_col_idx - destination_col_idx == -1):
      rotation_degree = 0 - current_degree
      final_degree = 0

  #case adjacent 5,6,7,8
  elif(abs(current_position_row_idx - destination_row_idx) == 1) & (abs(current_position_col_idx - destination_col_idx) == 1):
    #case 5
    if(current_position_row_idx - destination_row_idx == 1) & (current_position_col_idx - destination_col_idx == 1):
      rotation_degree = 135 - current_degree
      final_degree = 135
    #case 6
    elif(current_position_row_idx - destination_row_idx == 1) & (current_position_col_idx - destination_col_idx == -1):
      rotation_degree = 45 - current_degree
      final_degree = 45
    #case 7
    elif(current_position_row_idx - destination_row_idx == -1) & (current_position_col_idx - destination_col_idx == -1):
      rotation_degree = 315 - current_degree
      final_degree = 315
    #case 8
    elif(current_position_row_idx - destination_row_idx == -1) & (current_position_col_idx - destination_col_idx == 1):
      rotation_degree = 225 - current_degree
      final_degree = 225

  
 
  if rotation_degree >= 180:
    rotation_degree = rotation_degree - 360
  if rotation_degree <= -180:
    rotation_degree = rotation_degree + 360
  
  return [rotation_degree, final_degree]



def path2xy(r,c,path,unitcost):
  xy = np.zeros([len(path),2])

  for i in range(len(path)):
    row_idx = path[i] // c
    column_idx = path[i] % c
    
    xy[i,0] = (row_idx * unitcost) + (unitcost / 2)
    xy[i,1] = (column_idx * unitcost) + (unitcost / 2)

    return xy
  

#this function does not use Dji tello drone APIs and is only for testing purposes
def path2command_test(r,c,adj_matrix,path,init_degree,viualize_flag=True):
  #by considering a 3*4 graph as below:
  # 0  1  2  3
  # 4  5  6  7
  # 8  9 10 11
  #and also considering current position is point 5 
  #init_degree = 0 means that the direction is into point 6
  #and init_degree = 90 means that the direction is into point 1
  #and init_degree = 180 means that the direction is into point 4 ...

  height = np.zeros([len(path),1]) # to track tello height
  height[0] = 100 # to track tello height
  current_position = path[0]
  current_degree = init_degree

  for i in range(1,len(path)):
    destination = path[i]
    [rot_degree, current_degree] = calcul_rotation(r,c,current_position,current_degree,destination)
    
    print ('rotate ', rot_degree)
    print('move', int(adj_matrix[path[i-1],path[i]]))
    current_position = path[i]
    height[i] = 100 # to track tello height
    ####now we get the coordinates of the drone already landed with respect to the origin
    

  if viualize_flag:
    unitcost = np.min(np.min(adj_matrix))
    xy = path2xy(r,c,path,unitcost)
    xyz = np.append(xy,height,axis=1)
    xyz_df = pd.DataFrame(xyz, columns = ['X','Y','H'])

    fig = px.line_3d(xyz_df, x="X", y="Y", z="H", color='H')
    fig.show()

  return [current_position, current_degree,xy[-1]]

def path2command(r,c,adj_matrix,path,init_degree,viualize_flag=True):
  #by considering a 3*4 graph as below:
  # 0  1  2  3
  # 4  5  6  7
  # 8  9 10 11
  #and also considering current position is point 5 
  #init_degree = 0 means that the direction is into point 6
  #and init_degree = 90 means that the direction is into point 1
  #and init_degree = 180 means that the direction is into point 4 ...

  tello = Tello()
  tello.connect()
  print(f"Battery: {tello.get_battery()}")
  

  # Take off
  tello.takeoff()
  time.sleep(5)

  height = np.zeros([len(path),1]) # to track tello height
  height[0] = tello.get_height()# to track tello height
  print(height[0]) 
  current_position = path[0]
  current_degree = init_degree

  for i in range(1, len(path)):
    destination = path[i]
    [rot_degree, current_degree] = calcul_rotation(r, c, current_position, current_degree, destination)

    if i == len(path) - 1:
      if current_degree < 0 or current_degree > 180:
        current_degree = random.choice([0, 45, 90, 135, 180])
        rot_degree = current_degree - path[i-1]

    if rot_degree != 0:
        tello.rotate_counter_clockwise(rot_degree)
    
    tello.move_forward(int(adj_matrix[path[i-1], path[i]]))
    height[i] = tello.get_height()
    current_position = path[i]
        
    # Move forward one grid unit
    tello.move_forward(int(adj_matrix[path[i-1],path[i]]))

    print(f"the height is :{tello.get_height()}")
    print(f"the speed on the X axis is : {tello.get_speed_x()}")
  
    
    height[i] = tello.get_height() # to track tello height
    current_position = path[i]
    # print(f"current coordinate of camera : {path2xy(r,c,path,unitcost)}")
  
  if viualize_flag:
    unitcost = np.min(np.min(adj_matrix))
    xy = path2xy(r,c,path,unitcost)
    xyz = np.append(xy,height,axis=1)
    xyz_df = pd.DataFrame(xyz, columns = ['X','Y','H'])

    fig = px.line_3d(xyz_df, x="X", y="Y", z="H", color='H')
    fig.show()
  
  return [current_position, current_degree,xy[-1]]

##################################
#run the code
rows = 8
columns = 3
start = 0
goal = 15
#generate adj matrix:
adj_matrix = map_graph_generator(rows,columns,60,[4,8,10])#the size of tiles in cm is 60
# Define the file path


# Define the file path
path = os.path.expanduser('~')
directory = "/home/elham/thesis/drone_project_thesis/thesis_tello_navigation_astar"
file_path = os.path.join(path, directory, "adjmat_lav.csv")

# Save the array to the file
np.savetxt(file_path, adj_matrix, delimiter=',')

#apply A* to find shortest path
shortest_path = astar(adj_matrix, start, goal)
print(shortest_path)
#apply move:
[final_position, final_degree,xy] = path2command_test(rows,columns,adj_matrix,shortest_path,0)
print(final_degree)
print(final_position)
print(xy)
################################################################################

# #return:uncomment this part if you wish the drone to come back to its initial position
# shortest_path_return = astar(adj_matrix, final_position, start)
# print(shortest_path_return)
# #apply move:
# [final_position2, final_degree2] = path2command(rows,columns,adj_matrix,shortest_path_return,final_degree)

#3D visualization plot
user_home = os.path.expanduser('~')
directory = os.path.join(user_home,"thesis/drone_path_planning/thesis_tello_navigation_astar/path.txt")
data = np.loadtxt('/home/elham/thesis/drone_path_planning/thesis_tello_navigation_astar/path.txt')
ax = plt.figure().add_subplot(projection='3d')
ax.axes.set_xlim3d(left=-10, right=10) 
ax.axes.set_ylim3d(bottom=-10, top=10) 
ax.axes.set_zlim3d(bottom=-10, top=10)



for c in range(data.shape[0]):
    r = R.from_quat([data[c,3], data[c,4], data[c,5], data[c,6]])
    i = np.dot(r.as_dcm(), np.array([1, 0, 0]))
    j = np.dot(r.as_dcm(), np.array([0, 1, 0]))
    k = np.dot(r.as_dcm(), np.array([0, 0, 1]))
    x = data[c,0]
    y = data[c,1]
    z = data[c,2]

    ax.quiver(x, y, z, i[0], i[1], i[2], color='r', length=1.0, normalize=True)
    ax.quiver(x, y, z, j[0], j[1], j[2], color='g', length=1.0, normalize=True)
    ax.quiver(x, y, z, k[0], k[1], k[2], color='b', length=1.0, normalize=True)




ax.plot(data[:,0],data[:,1],data[:,2],'.-')

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()






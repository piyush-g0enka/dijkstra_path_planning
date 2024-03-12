#!/usr/bin/env python3
import numpy as dig
import cv2 as computer_vision
import math as ganit
from queue import PriorityQueue as PQ


class PathPlanning:

    def __init__(self):

        self.load_env()
        self.prompt_input()

        self.node_queue = PQ()
        self.node_queue.put((0, self.start_point))
        self.current_edge_cost = {self.start_point: 0}
        self.parent = {self.start_point: self.start_point}
        self.visited_list = [] 

        self.run_dijkstra()
        self.animate()


    # This fn loads the map with obstacles
    def load_env(self):

        self.width = 1200
        self.height = 500
        brightness = 255
        self.map = dig.ones((self.height, self.width, 3), dtype=dig.uint8) * brightness

        # list of all rectangle obstacles
        # 'rb' stands for bloated rectangles [bloated by 5 units]
        # 'r' stands for normal rectangles
        rectangle_obstacles = [
            ['rb', (1020, 45),  (1105, 455)], 
            ['rb', (895, 45),  (1105, 130)],  
            ['rb', (895, 370),  (1105, 455)], 
            ['rb', (270, 0), (355, 405)],  
            ['rb', (0, 0),  (1200, 5)], 
            ['rb', (95, 95),  (180, 500)], 
            ['rb', (0, 0),  (5, 500)],  
            ['rb', (1195, 0),  (1200, 500)], 
            ['rb', (0, 495),  (1200, 500)],
            ['r', (900, 375),  (1100, 450)],  
            ['r', (900, 50),  (1100, 125)],  
            ['r', (275, 0), (350, 400)],  
            ['r', (100, 100),  (175, 500)],  
            ['r', (1025, 50),  (1100, 450)] 
        ]

        # list containg hexagon obstacles
        hexagon_obstacles=[[(650, 250), 150]]

        # add obstacles to map
        for obs in rectangle_obstacles:
        
            corrected_p1 = self.normal_coords_to_cv2(obs[1][0], obs[1][1])
            corrected_p2 = self.normal_coords_to_cv2(obs[2][0], obs[2][1])

            if obs[0] == 'r':
                computer_vision.rectangle(self.map, corrected_p1, corrected_p2, (70,70,70), -1)

            else:
                computer_vision.rectangle(self.map, corrected_p1, corrected_p2, (255,90,90), -1)

        for obs in hexagon_obstacles:

            middle_point = self.normal_coords_to_cv2(obs[0][0], obs[0][1])

            side_lines = []
            number_of_lines = 6
            internal_angle = 60
            for line_no in range(0, number_of_lines):
                angle = ganit.pi*(internal_angle*line_no-30)/180.0
                x_line = middle_point[0] + ganit.cos(angle)*obs[1]
                y_line = middle_point[1] + ganit.sin(angle)*obs[1]
                side_lines.append((int(x_line), int(y_line)))

            computer_vision.fillPoly(self.map, [dig.array(side_lines)], (70,70,70))
            computer_vision.polylines(self.map, [dig.array(side_lines)], thickness=4, color=(255,90,90),isClosed=True)

 
    # This fn is used to convert co-ordinates to cv2 co-ordinates for retrival of pixel values
    def normal_coords_to_cv2(self,x_normal, y_normal):
        y_cv2 = (self.height -1)  - y_normal
        x_cv2 = x_normal
        return x_cv2, y_cv2

    # This fn prompts user for input
    def prompt_input(self):

        self.start_point=None
        self.goal_point=None

        while True:

            user_input = input("Please input the start point coordinates [x y] separated by a space: ")
            coords = user_input.split()
            if len(coords)!=2:
                print ("Invaid point! Try again...")
                
            else:
                coords[0]= int(coords[0])
                coords[1]=int(coords[1])
                x, y = self.normal_coords_to_cv2(coords[0], coords[1])

                if x<0 or x>=1200 or y<0 or y>=500 :
                    print ("Invaid point! Try again...")

                elif self.map[y][x][2]!=255:
                    print ("Invaid point! Try again...")
                else:
                    start_point=(coords[0],coords[1])
                    print("Start point recorded!")
                    break

        while True:

            user_input = input("Please input the goal point coordinates [x y] separated by a space: ")
            coords = user_input.split()
            if len(coords)!=2:
                print ("Invaid point! Try again...")
            else:     
                coords[0]= int(coords[0])
                coords[1]=int(coords[1])
                x, y = self.normal_coords_to_cv2(coords[0], coords[1])
                if x<0 or x>=1200 or y<0 or y>=500 :
                    print ("Invaid point! Try again...")
                elif self.map[y][x][2]!=255:
                    print ("Invaid point! Try again...")
                else:
                    goal_point=(coords[0],coords[1])
                    print("Goal point recorded!")
                    break   

        print ("Start point--> "+ str(start_point) )     
        print ("Goal point--> "+ str(goal_point) ) 

        self.start_point = start_point
        self.goal_point = goal_point  

        print ("Computing path...") 


    # Output valid adjacent nodes for a given node
    def compute_adjacent_nodes(self, node):

        x_node, y_node = node
        adjacent_nodes = []

        for adj_x in [-1,0,1]:
            for adj_y in [-1,0,1]:
                if adj_y == 0 and adj_x == 0:
                    continue
                adj_node = (x_node + adj_x, y_node + adj_y)
                xn, yn = self.normal_coords_to_cv2(adj_node[0], adj_node[1])
                if self.map[yn][xn][2]==255:
                    adjacent_nodes.append(adj_node)
        return adjacent_nodes

    # this fn generates the animation
    def animate(self):
        start_x, start_y = self.normal_coords_to_cv2(self.start_point[0], self.start_point[1])
        goal_x, goal_y = self.normal_coords_to_cv2(self.goal_point[0], self.goal_point[1])
        node_incr = 0

        # We display the node visit sequence
        for visited_node in self.visited_list:
            node_incr += 1
            xn,yn= self.normal_coords_to_cv2(visited_node[0],visited_node[1])
            self.map[yn, xn] = (150, 150, 150)

            computer_vision.circle(self.map, (start_x, start_y), 6, (50, 255, 50), -1)  
            computer_vision.circle(self.map, (goal_x, goal_y), 6, (50, 50, 255), -1) 
            # To speed up the animation we update frame every 10 nodes
            if node_incr % 10 == 0:
                computer_vision.imshow("Map", self.map)
                computer_vision.waitKey(1)
        
        # We display the path
        for node in self.robot_movement:
            xn,yn = self.normal_coords_to_cv2(node[0], node[1])
            computer_vision.circle(self.map, (xn, yn), 1, (255, 255, 0), -1) 
            computer_vision.imshow("Map", self.map)
            computer_vision.waitKey(1)

        computer_vision.imshow("Map", self.map)
        computer_vision.waitKey(0)
        computer_vision.destroyAllWindows()

    # This fn runs the dijkstra algorithm
    def run_dijkstra(self):

        infinite = float('inf')
        # Loop until queue has nodes
        while not self.node_queue.empty():

            present_total_cost, node = self.node_queue.get()
            curr_edge_cost  = self.current_edge_cost.get(node, infinite)

            # skip the node if node already optimised
            if present_total_cost > curr_edge_cost:
                continue
            
            # break if goal reached
            if node == self.goal_point:
                print("GOAL Reached!")
                break
            
            adjacent_nodes = self.compute_adjacent_nodes(node)
            for adjacent_node in adjacent_nodes:
                
                # calculate edge distance
                d = ganit.sqrt((node[0]-adjacent_node[0]) * (node[0]-adjacent_node[0]) + (node[1]-adjacent_node[1]) *(node[1]-adjacent_node[1]) )
                added_edge_cost = round(d,1)
                updated_edge_cost = self.current_edge_cost[node] + added_edge_cost

                # add/update adjacent node to our system
                if  (adjacent_node not in self.current_edge_cost) or  (updated_edge_cost < self.current_edge_cost[adjacent_node]) :
                    self.current_edge_cost[adjacent_node] = updated_edge_cost
                    lowest_edge_cost = updated_edge_cost
                    self.visited_list.append(adjacent_node)
                    self.node_queue.put((lowest_edge_cost, adjacent_node))
                    self.parent[adjacent_node] = node
                    
        # calculate path using backtracking
        self.robot_movement = []
        node = self.goal_point
        while node != self.start_point:
            self.robot_movement.append(node)
            node = self.parent[node]
        self.robot_movement.append(self.start_point)
        self.robot_movement.reverse()



if __name__ == "__main__":
    PathPlanning()
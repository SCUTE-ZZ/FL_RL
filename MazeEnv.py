import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self, n, m):
        self.maze_size = [n,m] 
        self.maze_map = []

        self.target_size = 2
        self.sapce_size = 2
        
        self.direction_space = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.move_list = [(-1,0),(1,0),(0,-1),(0,1)]
        self.move_wall = [-1,self.maze_size[0],-1,self.maze_size[1]]
        self.distance_space = 0
        self.state_space = 0
        
        self.target_num = 0
        self.target_pos = []
        self.space_pos = []
        self.cargo_pos = []

        self.generate_maze()
        self.state = self.get_state()

        # self.maze_data = self.generate_maze((self.maze_size, self.maze_size), self.target_size, self.space_size)
        # self.update_cargo_pos()
        # self.state()
        
        self.reward = {
            "hit_wall": -10.0, 
            "hit_space": -10.0, 
            "hit_target": -1,
            "destination": 1000.0,
            "finished": 10000.0,
            "default": 0,
            "state_score": 10.0
        }
    

    def generate_maze(self):
        """
            prim随机迷宫算法
            :param maze_size: 迷宫的宽度，生成迷宫尺寸为 maze_n * maze_m
            : maze_size // 5
            : target_size // 2
            : space_size // 2
            
        """
        # mark = np.zeros(maze_size, dtype=np.int)
        # maze_data = np.zeros(self.maze_size[0] * self.maze_size[1], dtype=np.int)
        
        random_numbers = random.sample(range(0, self.maze_size[0] * self.maze_size[1]), self.target_size + self.sapce_size)
        
        for i in range(self.target_size):
            if(random_numbers[i]//self.maze_size[0] != 0):
                self.target_num += 1
                self.target_pos.append((random_numbers[i]//self.maze_size[1], random_numbers[i]%self.maze_size[1]))
            else:
                self.target_pos.append((-1,-1))
        for i in range(self.target_size, self.target_size+self.sapce_size):
            self.space_pos.append((random_numbers[i]//self.maze_size[1], random_numbers[i]%self.maze_size[1]))
        
        # self.start_point = self.target_pos[0]
        # self.destination = self.space_pos
        # return maze_data
    
    def get_state(self):
        # maze_size 
        # target_pos  
        # space_pos
        return tuple(self.maze_size + [item for sublist in self.target_pos + self.space_pos for item in sublist])
      
    def get_n_states(self):
        return 2 +  self.target_size * 2 + self.sapce_size * 2
    
    def get_n_actions(self):
        return self.sapce_size * 2
    
    def get_state_score(self):
        score = 0
        for i in self.target_pos:
            if(i[0]==-1):
                continue
            score += i[0]
        return score
        
    def step_one(self, space_index, direction):
#         print(space_index, direction)
        reward = 0
        new_x = self.space_pos[space_index][0] + self.move_list[direction][0]
        new_y = self.space_pos[space_index][1] + self.move_list[direction][1]

        if(new_x<0 or new_y<0 or new_x>=self.maze_size[0] or new_y>=self.maze_size[1]):
            reward += self.reward['hit_wall']
            # print("hit_wall")
        elif((new_x,new_y) in self.space_pos): 
            reward += self.reward['hit_space']
            # print("hit_space")
            index = self.space_pos.index((new_x,new_y))
            self.space_pos[index] = self.space_pos[space_index]
            self.space_pos[space_index] = (new_x,new_y)
            
        elif((new_x,new_y) in self.target_pos):
            reward += self.reward['hit_target']
            # print("hit_target")
            index = self.target_pos.index((new_x,new_y))
            self.target_pos[index] = self.space_pos[space_index]
            if(self.target_pos[index][0]==0):
                reward += self.reward['destination']
                # print("destination")
                self.target_pos[index] = (-1, -1)
                self.target_num -= 1
                if(self.target_num == 0):
                    reward += self.reward['finished']
                    # print("finished")
            self.space_pos[space_index] = (new_x,new_y)
        else:
            reward += self.reward['default']
            # print("default")
            self.space_pos[space_index] = (new_x,new_y)

        return reward
    
    def show_action(self, action):
        direction=['^','v','<','>']
        new_action=[]
        for space_index in range(len(action)//2):
            step = self.get_step(action,space_index)
            new_action.append((direction[min(3,int(action[space_index*2]*4))], step))
        print(new_action)
                  
    def normal_direction(self, direction):
        return min(3,int(direction*4))
    
    def step(self, action):
        """
        Move the robot location according to its location and direction
        Return the new location and moving reward
        """
        reward = -1 - self.get_state_score() * self.reward['state_score']
        
        if(len(action) != len(self.space_pos) * 2):
            raise ValueError("Invalid Action")
        
        for space_index in range(len(action)//2):
            step = self.get_step(action,space_index)
            for j in range(step):
#                 print(action[space_index*2])
                reward += self.step_one(space_index, self.normal_direction(action[space_index*2]))
        self.state = self.get_state()
        
        reward += self.get_state_score() * self.reward['state_score']
        return self.state, reward, self.target_num == 0

    def get_step(self, action, space_index):
        max_step = 0
        direction = self.normal_direction(action[space_index*2])
        if(direction == 0):
            max_step = self.space_pos[space_index][0] + 1
        elif(direction == 1):
            max_step = self.maze_size[0] - self.space_pos[space_index][0]
        elif(direction == 2): 
            max_step = self.space_pos[space_index][1] + 1
        elif(direction == 3):
            max_step = self.maze_size[1] - self.space_pos[space_index][1]
        else:
            raise ValueError("Invalid Direction")
        return int(max_step * action[space_index*2+1])

    def reset(self, n = None , m = None):
        if(n == None):
            self.__init__(self.maze_size[0], self.maze_size[1])
            while(self.target_num == 0):
                self.__init__(self.maze_size[0], self.maze_size[1])
        else:
            self.__init__(n, m)
            while(self.target_num == 0):
                self.__init__(n, m)
        return self.state

    def random_action(self):
        action = []
        for i in range(len(self.space_pos)):
            direction = random.random()
            distance = random.random()
            action.append(direction)
            action.append(distance)
        return action
    
    def __repr__(self):
#         height, width, walls = self.maze_data.shape
        self.draw_maze()
#         self.draw_robot()

        plt.show()

#         return 'Maze of size (%d, %d)' % (height, width)
        return ''

    def draw_maze(self):
        grid_size = 1
        r, c = self.maze_size
        # 设置左上角为坐标原点
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.axis('off')
#         plt.plot(x,y)
        for i in range(r):  # 绘制墙壁
            for j in range(c):
                start_x = j * grid_size
                start_y = i * grid_size
                plt.hlines(start_y, start_x, start_x +
                           grid_size, color="black")
                plt.vlines(start_x + grid_size, start_y,
                           start_y + grid_size, color="black")
                plt.hlines(start_y + grid_size, start_x,
                           start_x + grid_size, color="black")
                plt.vlines(start_x, start_y, start_y +
                           grid_size, color="black")
#         # 绘制出入口

        for i in range(r):  # 绘制墙壁
            for j in range(c):
                rect_2 = plt.Rectangle([j,i], grid_size,
                                       grid_size, edgecolor=None, color="green")
                ax.add_patch(rect_2)
                
        for target_pos in self.target_pos:
            if(target_pos[0]==-1):
                continue
            rect_2 = plt.Rectangle(target_pos[::-1], grid_size,
                                   grid_size, edgecolor=None, color="red")
            ax.add_patch(rect_2)
            
        
        for i, space_pos in enumerate(self.space_pos):
            if(space_pos[0]==-1):
                continue
            rect_2 = plt.Rectangle(space_pos[::-1], grid_size,
                                   grid_size, edgecolor=None, color=(1.0-i*0.1,1.0-i*0.1,1.0-i*0.1))
            ax.add_patch(rect_2)
            
        plt.show()

# maze = MazeEnv(5, 5)
# print(maze.get_state())
# maze.draw_maze()
# action = maze.random_action()
# maze.move(action)
# print(action)
# maze.draw_maze()


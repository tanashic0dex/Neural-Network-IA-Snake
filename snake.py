import numpy as np 
from collections import deque
from random import randint


#BGR
YELLOW = (0,255,255)
GREY = (100,100,100)
BODY_COLOR = (0,0,0) #BLACK

class Snake:
    """
    Represents the snake.
    """
    def __init__(self, x_start, y_start, env_width, env_height):
        """
        Snake params.

        :param x_start: start x position.
        :type x_start: int.
        :param y_start: start y position
        :type y_start: int.
        :param env_width: environment width.
        :type env_width: int.            
        :param env_height: environment height.
        :type env_height: int.
        """
        self.ENV_WIDTH = env_width
        self.ENV_HEIGHT = env_height
        
        #Generating random directions for initiating the snake
        a = randint(0,3)
        init_dir = 0 
        
        
        if a == 0:
            self.X = deque([x_start, x_start-1, x_start -2, x_start-3])
            self.Y = deque([y_start, y_start, y_start, y_start])
            init_dir = 1
        elif a == 1:
            self.X = deque([x_start, x_start+1, x_start+2, x_start+3])
            self.Y = deque([y_start, y_start, y_start, y_start])
            init_dir = 0
        elif a == 2:
            self.Y = deque([y_start, y_start-1, y_start-2, y_start-3])
            self.X = deque([x_start, x_start, x_start, x_start])
            init_dir = 3
        else:
            self.Y = deque([y_start, y_start+1, y_start+2, y_start+3])
            self.X = deque([x_start, x_start, x_start, x_start])
            init_dir = 2
       
        self.is_alive = True 
        self.MOVES = max(100, env_width*env_height)
        self.LENGTH = 4
        self.INITIAL_DIRECTION = init_dir


    def draw(self, env):
        """
        Draw snake on the matrix.

        :param env: represents the environment of the game.
        :type env: Class Snake_Env().
        """
        for body_y, body_x in zip(self.Y, self.X):
            env[body_y, body_x] = BODY_COLOR

        head_x, head_y = self.head_pos()
        env[head_y, head_x] = GREY if self.is_alive else YELLOW 
        return env 
            
    
    def eat_food(self, x_food, y_food):
        """
        Update snake's parameters if it eats food.

        :param x_food: food x position.
        :type x_food: int.
        :param y_food: food y position.
        :type y_food: int.
        """
        self.X.appendleft(x_food) 
        self.Y.appendleft(y_food)
        self.LENGTH += 1
        self.MOVES += 100

        
    def update(self, x_change, y_change):
        """
        Move the snake forward.

        :param x_change: x position change value.
        :type x_change: int.
        :param y_change: y position change value.
        :type y_change: int.
        """
        #Updating snake's body position
        for i in range(self.LENGTH-1,0,-1):
            self.X[i] = self.X[i-1]
            self.Y[i] = self.Y[i-1]

        #Updating head position
        self.X[0] = self.X[0] + x_change
        self.Y[0] = self.Y[0] + y_change

        self.MOVES -= 1


    def bit_itself(self):
        """
        Checks if the snake bit itself. 
        """
        for i in range(1,self.LENGTH):
            if self.X[0] == self.X[i] and self.Y[0] == self.Y[i]:
                return True
        return False

    
    def is_on_body(self, check_x, check_y, remove_last=True):
        """
        Check if a position is on the body of the snake.

        :param check_x: x position to be checked.
        :type check_x: int
        :param check_y: y position to be checked.
        :type check_y: int 
        :param remove_last: remove the checked position from the lists. 
        :type remove_last: boolean
        """
        X, Y = self.X.copy(), self.Y.copy()
        
        if remove_last:    #don't consider the last tail part
            X.pop()
            Y.pop()
        for x,y in zip(X,Y):
            if x == check_x and y == check_y:
                return True
        return False


    def head_pos(self):
        """
        Returns the head coordinates.
        :rtype: int, int.
        """
        return self.X[0], self.Y[0]


    def kill(self):
        """
        Update when the snake dies.
        """
        self.is_alive = False

    
    def set_length(self, length):
        """
        Update the lenght of the snake.

        :param length: length of the snake.
        :type length: int.
        """
        self.LENGTH = length
    

    def look(self, x_food, y_food, boundaries):
        """
        Look in all directions.

        :param x_food: food x position.
        :type x_food: int.
        :param y_food: food y position.
        :type y_food: int.
        :param boundaries: boundaries of the game.
        :type boundaries: list of int with dimension (1, 4).
        :return: state of the environment.
        :rtype: NumPy array with dimension (1, 18).
        """
        #Look up
        up = self.lookInDirection(boundaries, y=-1, x=0)
        #Look up/right
        up_right = self.lookInDirection(boundaries, y=-1, x=1)
        #Look right
        right = self.lookInDirection(boundaries, y=0, x=1)
        #Look down/right
        down_right = self.lookInDirection(boundaries, y=1, x=1)
        #Look down
        down = self.lookInDirection(boundaries, y=1, x=0)
        #Look down/left
        down_left = self.lookInDirection(boundaries, y=1, x=-1)
        #Look left
        left = self.lookInDirection(boundaries, y=0, x=-1)
        #Look up/left
        up_left = self.lookInDirection(boundaries, y=-1, x=-1)

        head_x, head_y = self.head_pos()
        food_position = np.array([head_x - x_food, head_y - y_food])
        return np.hstack((food_position, up, up_right, right, down_right, down, down_left, left, up_left))
        #array of size 2*8 + 2 = 18
	

    def lookInDirection(self, boundaries, y, x):
        """
        Look in a specific direction.

        :param boundaries: boundaries of the game.
        :type boundaries: list of int with dimension (1, 4).
        :param y: direction to look further in y-axis. 
        :type y: int.
        :param x: direction look further in x-axis.
        :type x: int.
        :return: NumPy array with the value of 1/distance to the wall or tail and 
                    distance to the tail if it is in the direction.
        :rtype: NumPy array with dimension (1, 2).
        """
        tail_distance = 0   # 0 if tail is not in this direction
        distance = 1        #starting with 1 as it is the min distance of wall

        curr_x, curr_y = self.head_pos()
        check_x, check_y = curr_x + x, curr_y + y       #look one step further in the direction
        x1, x2, y1, y2 = boundaries

        while check_y >= y1 and check_y <= y2 and check_x >= x1 and check_x <= x2:

            if tail_distance==0 and (self.is_on_body(check_x, check_y, remove_last = False)):
                tail_distance = (1/distance)

            check_y += y
            check_x += x
            
            distance += 1
        
        return np.array([1/distance, tail_distance])
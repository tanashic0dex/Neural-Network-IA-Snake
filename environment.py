from random import randint
from PIL import Image
import cv2
import numpy as np

from snake import Snake
from collections import deque
from utils import add_info

#BGR
WHITE = (255,255,255)
FOOD_COLOR = (0,0,255)      #RED
BOUNDARY_COLOR = (0,0,0)    #BLACK

class Snake_Env():
    """
    Represents the environment of the game.
    """
    def __init__(self, max_width, max_height, init_width, init_height, display_width, display_height):
        """
        Environment params.
        
        :param max_width: maximum width of the game border.
        :type max_width: int.
        :param max_height: max height of the game border.
        :type max_height: int.
        :param init_width: initial with of the game border.
        :type init_widht: int.
        :param init_height: initial height of the game border.
        :type init_height: int.
        :param display_width: width of the display box.
        :type display_width: int.
        :param display_height: height of the display box.
        :type display_height: int.
        """
        self.MAX_WIDTH = max_width
        self.MAX_HEIGHT = max_height
        self.WIDTH = init_width
        self.HEIGHT = init_height
        self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT = display_width, display_height
        self.STATE_SPACE = 18 
        self.ACTION_SPACE = 4
                

    def change_size(self, width_change, height_change):
        """
        Increase the size of the environment.

        :param width_change: width border increment while training.
        :type width_change: int. 
        :param height_change: height border increment while training.
        :type height_change: int.
        """
        self.WIDTH = min(self.WIDTH + width_change, self.MAX_WIDTH-2)
        self.HEIGHT = min(self.HEIGHT + height_change, self.MAX_HEIGHT-2)

            
    def get_boundaries(self):
        """
        Get boundaries of the game.
        """
        odd_w, odd_h = self.WIDTH%2, self.HEIGHT%2
        mid_width, mid_height = int(self.MAX_WIDTH/2), int(self.MAX_HEIGHT/2)
        x1, x2 = (mid_width-int(self.WIDTH/2), mid_height-1+int(self.WIDTH/2)+odd_w) 
        y1, y2 = (mid_width-int(self.HEIGHT/2), mid_height-1+int(self.HEIGHT/2)+odd_h)
        return x1, x2, y1, y2


    def get_randoms(self, length = 4):
        """
        Returns random positions within the boundary.

        :param length: length of the snake.
        :type length: int.
        """
        x1, x2, y1, y2 = self.get_boundaries()
        
        #Logic to keep the snake inside the boundary
        if length>1:
            max_w, max_h = self.MAX_WIDTH-length, self.MAX_HEIGHT-length
            if x1<length-1: x1 = length-1
            if x2>max_w: x2 = max_w
            if y1<length-1: y1 = length-1
            if y2>max_h: y2 = max_h    
        a = randint(x1, x2)
        b = randint(y1, y2)
        return a,b  


    def play_region(self, env):
        """
        Color the playable region.

        :param env: game environment.
        :type env: Class Snake_Env().
        """
        x1, x2, y1, y2 = self.get_boundaries()
        env[y1:y2+1, x1:x2+1, :] = WHITE
        return env


    def reset(self):
        """
        Reset the environment. Initializes the snake and the food.

        :return: state of the environment.
        :rtype: NumPy array with dimension (1, 18).
        """
        snake_x, snake_y = self.get_randoms(length=4)  
        self.SNAKE = Snake(snake_x, snake_y , self.WIDTH, self.HEIGHT)
        self.VELOCITY = self.SNAKE.INITIAL_DIRECTION
        self.FOOD_X, self.FOOD_Y = self.get_randoms()
        self.STATE =  self.SNAKE.look(self.FOOD_X, self.FOOD_Y, self.get_boundaries())

        return self.STATE

    
    def render(self, action):
        """
        Render the environment.

        :param action: action chosen by the agent.
        :type action: int.
        """
        env = np.zeros((self.MAX_HEIGHT, self.MAX_WIDTH, 3), dtype = np.uint8) #image with 3 channels RGB
        env = self.play_region(env)                                            #add allowed boundary
        env[self.FOOD_Y, self.FOOD_X] = FOOD_COLOR                             #add the food
        env = self.SNAKE.draw(env)                                             #add the snake 
        
        img = add_info(self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT, env, action) 
        cv2.imshow("Snake Game", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  #when Q is pressed
            print('Stop Execution')
            cv2.destroyAllWindows()
            quit()
        

    def step(self,action):
        """
        Agent takes an action and changes the environment.

        :param action: action chosen by the agent.
        :type action: int.
        """
        reward = -1.0
        info = 0        #0:nothing, bite:1, eat:2, out_boundary:3, moves_done:4 
        done, food_eaten = False, False
        update = False

        x_change, y_change = 0, 0

        #Logic to decide the action by the environment
        if action == 0:
            if self.VELOCITY == 1:   #if moving right
                x_change = 1 
            else:
                x_change = -1        #move left
                update = True
        elif action == 1:
            if self.VELOCITY == 0:   #if moving left
                x_change = -1 
            else:
                x_change = 1         #move right
                update = True
        elif action == 2:
            if self.VELOCITY == 3:   #if moving down
                y_change = 1 
            else:
                y_change = -1        #move up
                update = True
        elif action == 3:
            if self.VELOCITY == 2:   #if moving up
                y_change = -1 
            else:
                y_change = 1         #move down
                update = True
        
        if update:
            self.VELOCITY = action
        head_x, head_y = self.SNAKE.head_pos()
        new_head_x, new_head_y = head_x + x_change, head_y + y_change   #new head position
        
        #Check if the snake bit itself
        if self.SNAKE.is_on_body(new_head_x, new_head_y) and self.SNAKE.LENGTH > 2:   
            reward = -50
            done = True
            self.SNAKE.kill()
            info = 1
        
        #Check if it ate food
        if new_head_x == self.FOOD_X and new_head_y == self.FOOD_Y:
            self.SNAKE.eat_food(self.FOOD_X, self.FOOD_Y)                                                                         
            reward = 50
            food_eaten = True
            self.FOOD_X, self.FOOD_Y = self.get_randoms()
            info = 2
    
        #Check if the snake hit the wall
        x1, x2, y1, y2 = self.get_boundaries()
        if new_head_x < x1 or new_head_x > x2 or new_head_y < y1 or new_head_y > y2:
            reward = -50
            done = True
            self.SNAKE.kill()
            info = 3
        
        #Update the snake's position
        if not done and not food_eaten:
            self.SNAKE.update(x_change, y_change)
          
        if self.SNAKE.MOVES == 0:   #Limit of moves reached
            done = True
            info = 4
        
        boundaries = [x1, x2, y1, y2]
        self.STATE =  self.SNAKE.look(self.FOOD_X, self.FOOD_Y, boundaries)

        return self.STATE, reward, done, info
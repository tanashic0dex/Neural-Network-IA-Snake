import numpy as np
from PIL import Image

#BGR
GREEN = (0,255,0)
RED = (0,0,255)


def add_info(width, height, env, action):
    """
    Display the game as an image.

    :param width: width of the game border.
    :type width: int.
    :param height: height of the game border.
    :type height: int.
    :param env: game environment.
    :type env: Class Snake_Env().
    :param action: action chose by the agent.
    :type action: int.
    """
    extra_cols = 10   #extra columns for the buttons. See display_action() below.
    rows, cols, channels = env.shape
    display_matrix = np.ones((rows, cols+extra_cols, channels), dtype=np.uint8)*255  #white background
    display_matrix[:rows, 1:cols+1, :] = env 
    display_matrix = display_action(display_matrix, action, rows)
    
    img = Image.fromarray(display_matrix, 'RGB')
    scale = int(width/cols)
    img = np.array(img.resize((width + extra_cols*scale, height)))
        
    return img

def display_action(matrix, action, max_height):
    """
    Shows the action chosen by the snake as a green button.

    :param action: action chosen by the agent.
    :type action: int.
    :param max_height: max height of the game border.
    :type max_height: int.
    """
    mid_height = int(max_height/2)
    matrix[mid_height,-5,:] = RED    #left
    matrix[mid_height,-7,:] = RED    #right
    matrix[mid_height-1,-6,:] = RED  #up
    matrix[mid_height+1,-6,:] = RED  #down
    
    if action == 0:
        matrix[mid_height,-7,:] = GREEN    #left 
    elif action == 1:
        matrix[mid_height,-5,:] = GREEN    #right
    elif action == 2:
        matrix[mid_height-1,-6,:] = GREEN  #up 
    elif action ==3:
        matrix[mid_height+1,-6,:] = GREEN  #down 

    return matrix
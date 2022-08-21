from asyncio.mixins import _global_lock
import os
import numpy as np
import gym
import gym_minigrid
import pickle
import matplotlib.pyplot as plt
import imageio
import random

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def terminalCost(X, goal_coord):
    '''
    Generates terminal cost
    for environment
    _______________________
    returns the terminal cost for every state
    '''
    q = np.ones(X.shape)*np.inf
    # Part A
    if len(goal_coord) == 2:
        q[goal_coord[0],goal_coord[1]] = 0        
    # Part B
    else: 
        for n in range(0,len(goal_coord)):
            q[goal_coord[n][0],goal_coord[n][1],:,:,:,n,:,:] = 0
   
    return q

def invalidAction_A(x,u,info,walls):
    '''
    Checks if a combination of 
    state and action are valid for Part A.
    __________________________
    returns true if combination is invalid
    returns false if combination is valid
    '''
    # if on border square
    if x[0] == info['height']-1 or x[1] == info['height']-1 or x[0]==0 or x[1]==0:
        case = True # all border squares are walls in the environments
    # if in closed door
    elif x[3] == 0 and all([x[0],x[1]] == info['door_pos']):
        case = True
    # if attempting to pick up key
    elif u == 3:
        # Has the key already been picked up?
        key_x = info['key_pos'][0] # define indices of key as this to make simpler
        key_y = info['key_pos'][1]
        if x[4] == 1:
            case = True
        # Adjacent on cardinals
        elif [x[0],x[1]] == [key_x,key_y+1]: # below
            if   x[2] == 1:
                case = False # agent is facing up
            else:
                case = True
        elif [x[0],x[1]] == [key_x,key_y-1]: # above
            if   x[2] == 3:
                case = False # agent is facing down
            else:
                case = True
        elif [x[0],x[1]] == [key_x+1,key_y]: # right
            if   x[2] == 0:
                case = False # agent is facing left
            else:
                case = True
        elif [x[0],x[1]] == [key_x-1,key_y]: # left
            if   x[2] == 2:
                case = False # agent is facing right
            else:
                case = True           
        # Not adjacent on cardinals
        else: 
            case = True
    # if attempting to open door
    elif u == 4:
        # agent not holding key
        if x[4] != 1:
            case = True # agent was not holding key and attempted to open door
        # door already open
        elif x[3] == 1:
            case = True # agent tried to open door but was already open
        # Adjacent on left (never optimal to open door from right)
        elif [x[0],x[1]] == [info['door_pos'][0]-1,info['door_pos'][1]]:
            # check if oriented correctly
            if   x[2] == 2 and info['door_pos'][0] > x[0]:
                case = False # agent is facing right and door on right
            else:
                case = True  # agent not facing proper direction
        # Not adjacent on cardinals
        else: 
            case = True
    # if in a wall
    elif [x[0],x[1]] in walls:
        case = True
    else:
        case = False
    return case
 
def invalidAction_B(x,u,info,walls,key_loc,goal_loc):
    '''
    Checks if a combination of 
    state and action are valid for Part B.
    __________________________
    returns true if combination is invalid
    returns false if combination is valid
    '''
    #key_loc = {0:[1,1],1:[2,3],2:[1,6]} for reference

    # if on border square
    if x[0] == info['height']-1 or x[1] == info['height']-1 or x[0]==0 or x[1]==0:
        case = True # all border squares are walls in the environments
    # if in door 1 and it is closed
    elif x[3] == 0 and [x[0],x[1]] == [4,2]:
        case = True
    # if in door 2 and it is closed
    elif x[4] == 0 and [x[0],x[1]] == [4,5]:
        case = True
    # if attempting to pick up key
    elif u == 3:
        key_coords = np.asarray(key_loc[x[6]]) # define indices of key as this to make simpler
        agent_coords = np.asarray([x[0],x[1]])
        # Has the key already been picked up?
        if x[7] == 1:
            case = True
        # Adjacent on cardinals
        elif all(agent_coords == key_coords+[0,1]): # below
            if   x[2] == 1:
                case = False # agent is facing up
            else:
                case = True          
        elif all(agent_coords == key_coords+[0,-1]): # above
            if   x[2] == 3:
                case = False # agent is facing down
            else:
                case = True
        elif all(agent_coords == key_coords + [1,0]): # right
            if   x[2] == 0:
                case = False # agent is facing left
            else:
                case = True
        elif all(agent_coords == key_coords + [-1,0]): # left
            if   x[2] == 2:
                case = False # agent is facing right
            else:
                case = True              
        # Not adjacent on cardinals
        else: 
            case = True
    # if attempting to open door
    elif u == 4:
        # agent not holding key
        if x[7] != 1:
            case = True # agent was not holding key and attempted to open a door
        # Adjacent on left of door 1 (never optimal to open door from right)
        elif [x[0],x[1]] == [3,2]:
            # check if door 1 already open
            if x[3] == 1:
                case = True
            # door closed, now check if oriented correctly
            elif x[2] == 2:
                # check if facing right
                case = False # agent is facing right and door on right
            else:
                case = True  # agent not facing proper direction
        # Adjacent on left of door 2 (never optimal to open door from right)
        elif [x[0],x[1]] == [3,5]:
            # check if door 2 already open
            if x[4] == 1:
                case = True
            # door closed, now check if oriented correctly
            elif x[2] == 2:
                # check if facing right
                case = False # agent is facing right and door on right
            else:
                case = True  # agent not facing proper direction
        else: # not adjacent to door
            case = True
    # if in a wall
    elif [x[0],x[1]] in walls:
        case = True
    else:
        case = False
    return case
 
def stageCost_A(x,u,info,U,walls):
    '''
    Assigns cost to action for
    combination of current state and
    control input.
    
    If action is invalid assings inf,
    otherwise assigns associated control
    cost from U
    ________________________________
    returns l, the cost of x,u
    '''    
    if invalidAction_A(x,u,info,walls) == True:
        # Invalid Action
        l = np.inf
    else:
        # Valid Action
        l = U[u]
    return l

def stageCost_B(x,u,info,U,walls,key_loc,goal_loc):
    '''
    Assigns cost to action for
    combination of current state and
    control input.
    
    If action is invalid assings inf,
    otherwise assigns associated control
    cost from U
    ________________________________
    returns l, the cost of x,u
    '''    
    if invalidAction_B(x,u,info,walls,key_loc,goal_loc) == True:
        # Invalid Action
        l = np.inf
    else:
        # Valid Action
        l = U[u]
    return l

def motionModel_A(x,u,info,walls):
    '''
    Alters the state vector
    to next state based on control.
    
    If the control would have the 
    state move to an invalid state,
    the state remains the same.
    _______________________________
    returns x, the altered state
    '''
    x_p = [x[0],x[1],x[2],x[3],x[4]]
    # If invalid action, stay at current state
    if invalidAction_A(x,u,info,walls) == True:
        # keep state the same
        pass
    # Already in Goal
    elif [x_p[0],x_p[1]] == [info['goal_pos'][0],info['goal_pos'][1]]:
        # No reason to move
        pass
    # Valid Action
    else:
        # Move Forward
        if u == 0:
            # Move Left
            if x[2] == 0:
                x_p[0] = x_p[0] - 1
            # Move Up
            elif x[2] == 1:
                x_p[1] = x_p[1] - 1
            # Move Right
            elif x[2] == 2:
                x_p[0] = x_p[0] + 1              
            # Move Down
            elif x[2] == 3:
                x_p[1] = x_p[1] + 1
        # Turn Left (counter-clockwise)
        elif u == 1:
            if x[2] == 0: # decrementing by 1 would be outside index range
                x_p[2] = 3
            else:
                x_p[2] = x_p[2] - 1 # decrement by 1
        # Turn Right (clockwise)
        elif u == 2:
            if x[2] == 3:   # incrementing by 1 would be outside index range
                x_p[2] = 0
            else:
                x_p[2] = x_p[2] + 1 # increment by 1
        # Pickup Key
        elif u == 3:
            x_p[4] = 1 # 0 means key is still in env
        # Open Door
        elif u == 4:
            x_p[3] = 1 # 0 is closed door state
    # return modified state    
    return x_p
    
def motionModel_B(x,u,info,walls,key_loc,goal_loc):
    '''
    Alters the state vector
    to next state based on control.
    
    If the control would have the 
    state move to an invalid state,
    the state remains the same.
    _______________________________
    returns x, the altered state
    '''
    x_p = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
    
    # If invalid action, stay at current state
    if invalidAction_B(x,u,info,walls,key_loc,goal_loc) == True:
        # keep state the same
        pass        
    # already in any of the goal locations 
    elif [x_p[0],x_p[1]] == [5,1] or [x_p[0],x_p[1]] == [6,3] or [x_p[0],x_p[1]] == [5,6]:
        # No reason to move
        '''
        THIS MIGHT CAUSE PROBLEMS IN SOME SCENARIOS 
        (if passing through one goal to get to another would be optimal)
        '''
        pass
    # Valid Action
    else:
        # Move Forward
        if u == 0:
            # Move Left
            if x[2] == 0:
                x_p[0] = x_p[0] - 1
            # Move Up
            elif x[2] == 1:
                x_p[1] = x_p[1] - 1
            # Move Right
            elif x[2] == 2:
                x_p[0] = x_p[0] + 1              
            # Move Down
            elif x[2] == 3:
                x_p[1] = x_p[1] + 1
        # Turn Left (counter-clockwise)
        elif u == 1:
            if x[2] == 0: # decrementing by 1 would be outside index range
                x_p[2] = 3
            else:
                x_p[2] = x_p[2] - 1 # decrement by 1
        # Turn Right (clockwise)
        elif u == 2:
            if x[2] == 3:   # incrementing by 1 would be outside index range
                x_p[2] = 0
            else:
                x_p[2] = x_p[2] + 1 # increment by 1
        # Pickup Key
        elif u == 3:
            x_p[7] = 1 # 0 means key is still in env
        # Open Door
        elif u == 4:
            if x[1] == 2:   # next to door 1
                x_p[3] = 1  # open door 1
            else:           # next to door 2
                x_p[4] = 1  # open door 2
    # return modified state    
    return x_p
    
def step_cost(action):
    '''
    Returns the cost of an associated action
    ----------------------------------
    '''
    # Costs dictionary
    costs = {
        0: 3,
        1: 3,
        2: 3,
        3: 1,
        4: 1
        }

    return costs[action] # the cost of action

def step(env, action):
    '''
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    '''
    actions = {
        0: env.actions.forward,
        1: env.actions.left,
        2: env.actions.right,
        3: env.actions.pickup,
        4: env.actions.toggle
        }

    _, _, done, _ = env.step(actions[action])
    return step_cost(action), done

def generate_random_env(seed, task):
    ''' 
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    '''
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task)
    env.seed(seed)
    env.reset()
    return env

def load_env(path):
    '''
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    '''
    with open(path, 'rb') as f:
        env = pickle.load(f)
    
    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec
        }
    
    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i),
                          gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Door):
                info['door_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])    
            
    return env, info

def load_random_env(env_folder):
    '''
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    '''
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
    env_path = random.choice(env_list)
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
    
    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec,
        'door_pos': [],
        'door_open': [],
        }
    
    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i),
                          gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Door):
                info['door_pos'].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info['door_open'].append(True)
                else:
                    info['door_open'].append(False)
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])    
            
    return env, info, env_path

def save_env(env, path):
    with open(path, 'wb') as f:
        pickle.dump(env, f)

def plot_env(env):
    '''
    Plot current environment
    ----------------------------------
    '''
    img = env.render('rgb_array', tile_size=32)
    plt.figure()
    plt.imshow(img)
    plt.show()

def draw_gif_from_seq(seq, env, path='./gif/doorkey.gif'):
    '''
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]
    
    env:
        The doorkey environment
    '''
    with imageio.get_writer(path, mode='I', duration=0.8) as writer:
        img = env.render('rgb_array', tile_size=32)
        writer.append_data(img)
        for act in seq:
            img = env.render('rgb_array', tile_size=32)
            step(env, act)
            writer.append_data(img)
    print('GIF is written to {}'.format(path))
    return
    

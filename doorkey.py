import numpy as np
import gym
from utils import *

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

# Used to create policy as a list of str instead of int
policyStrTable = {0:'MF',1:'TL',2:'TR',3:'PK',4:'UD'}

# Environments for part A
envDict = {'./envs/doorkey-5x5-normal.env':1,
            './envs/doorkey-6x6-direct.env':2,
            './envs/doorkey-6x6-normal.env':3,
            './envs/doorkey-6x6-shortcut.env':4,
            './envs/doorkey-8x8-direct.env':5,
            './envs/doorkey-8x8-normal.env':6,
            './envs/doorkey-8x8-shortcut.env':7}

'''
STATES A
x1 = {0,...,h-1} x_coord
x2 = {0,...,h-1} y_coord
x3 = {0,1,2,3} dir
x4 = {0,1} door
x5 = {0,1} key

facing_left =   0
facing_up   =   1
facing_right=   2
facing_down =   3

door_closed = 0
door_open = 1

key_on_map = 0
key_obtained = 1
'''

def doorkey_problem_A(envNum, info):
    '''
    Finds the optimaal path in any of the following environments:
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
    ____________________________
    returns the optimal policy and the associated value function sequence
    '''
    # Control Keywords
    MF = 0 # Move Forward
    TL = 1 # Turn Left
    TR = 2 # Turn Right
    PK = 3 # Pickup Key
    UD = 4 # Unlock Door

    policyTable = {0:MF,1:TL,2:TR,3:PK,4:UD}
    
    # GRID WIDTH AND HEIGHT
    h = info['height']

    init_agent_pos = info['init_agent_pos']
    init_agent_dir = info['init_agent_dir']

    # Coordinate Transformation - Directions
    if all(init_agent_dir == [-1,0]):
        init_agent_dir = 0
    elif all(init_agent_dir == [0,-1]):
        init_agent_dir = 1
    elif all(init_agent_dir == [1,0]):
        init_agent_dir = 2
    elif all(init_agent_dir == [0,1]):
        init_agent_dir = 3

    # Initial States for A
    x01 = init_agent_pos[0]
    x02 = init_agent_pos[1]
    x03 = init_agent_dir
    x04 = 0 
    x05 = 0 

    # Define initial state
    x0 = (x01,x02,x03,x04,x05)
    print('Initial State:')
    print(x0)

    # TIME HORIZON
    T = 200 # some arbitrarily large amount

    # STATE SPACE
    X = np.zeros((h,h,4,2,2))

    # VALUE FUNCTION
    V = np.ones((h,h,4,2,2,T))*np.inf   

    # OPTIMAL CONTROL POLICY
    optim_act_seq = {}

    # CONTROL SPACE - ('ACTION',COST)
    U = dict([(MF,3),(TL,3),(TR,3),(PK,1),(UD,1)])
    
    # TERMINAL COST    
    q = terminalCost(X, info['goal_pos'])

    # Part A Coordinates
    
    # Walls env:coordinates
    walls =    {1:[[2,1],[2,3]],
                2:[[2,3],[3,2],[3,3]],
                3:[[3,4],[3,2],[3,3]],
                4:[[2,3],[3,2],[3,3]],
                5:[[1,3],[2,3],[4,2],[4,3],[4,4],[4,5]],
                6:[[1,3],[2,3],[4,2],[4,3],[4,4],[4,5],[4,6]],
                7:[[1,3],[2,3],[2,5],[3,5],[4,2],[4,3],[4,4],[4,5]]}

    # Terminal Time Value Function
    V[:,:,:,:,:,T-1] = q
    Q = np.zeros(5)

    for t in range(T-1,0,-1):  # loop over time

        # Loop over X
        for x in np.ndindex(np.shape(X)):

            # Loop over U
            for u in range(0,5):
                
                # Compute x_prime (x_t+1)
                x_p = motionModel_A(x,u,info,walls[envNum])

                # Define Cost Function for each different action
                Q[u] = stageCost_A(x,u,info,U,walls[envNum]) + V[x_p[0],x_p[1],x_p[2],x_p[3],x_p[4],t]

            # Define Value Function for each state x at time t
            # Only update a state's value function if less than past value function
            V[x][t-1] = min(Q) if V[x][t] > min(Q) else V[x][t]
    
        # When all value functions are the same as prior iteration, terminate loop
        if np.array_equal(V[:,:,:,:,:,t-1],V[:,:,:,:,:,t]) and t != T-1:
            print('All Values Converged at time:')
            print(t)
            break

    # Generating Optimal Policy starting from x0
    optim_act_seq = []
    seq_Value = []
    curState = x0
    while V[curState][t-1] > 0: # look at values at terminal time
        # check each control action and return one that provides minimum cost
        seq_Value.append(V[curState][t-1])

        for u in range(0,5):
            x_cand= motionModel_A(curState,u,info,walls[envNum])
            newState = (x_cand[0],x_cand[1],x_cand[2],x_cand[3],x_cand[4])
            Q[u] = V[newState][t-1] # compute cost at all future states with control         
        opt_act = np.argmin(Q)
        optim_act_seq.append(opt_act) # add optimal action to sequence       
        A = motionModel_A(curState,opt_act,info,walls[envNum]) # move to next state
        curState = (A[0],A[1],A[2],A[3],A[4])

    seq_Value.append(V[curState][t-1])        
    
    policy = []
    # Translate sequence to strings
    for n in range(0,len(optim_act_seq)):
        # policy for initial state
        policy.append(policyTable[optim_act_seq[n]])

    return policy, seq_Value

def doorkey_problem_B(env, info):
    '''
    Finds the optimal path for any random environment
    _________________________________________________
    returns optimal policy and value function sequence for the env
    '''
    # Control Keywords
    MF = 0 # Move Forward
    TL = 1 # Turn Left
    TR = 2 # Turn Right
    PK = 3 # Pickup Key
    UD = 4 # Unlock Door

    policyTable = {0:MF,1:TL,2:TR,3:PK,4:UD}
    
    # GRID WIDTH AND HEIGHT
    h = 8

    # GIVEN
    init_agent_pos = [3,5]
    init_agent_dir = 1      # facing up

    goal_loc = {0:[5,1],1:[6,3],2:[5,6]}
    key_loc = {0:[1,1],1:[2,3],2:[1,6]}

    # reverse search the dictionary
    goal_key = next(key for key, value in goal_loc.items() if np.array_equal(np.array(value),info['goal_pos']))
    key_key = next(key for key, value in key_loc.items() if np.array_equal(np.array(value),info['key_pos']))

    door1_state = 0
    door2_state = 0
    # Defining initial states for the doors
    if env.grid.get(4,2).is_open:
        door1_state = 1
    if env.grid.get(4,5).is_open:
        door2_state = 1

    # Generating Initial States for B
    x01 = init_agent_pos[0]             # agents x-coord {0,...,h}
    x02 = init_agent_pos[1]             # agents y-coord {0,...,h}
    x03 = init_agent_dir                # agents direction {0,1,2,3} = {L,U,R,D}
    x04 = door1_state                   # door 1 state {0,1} = {closed,open}
    x05 = door2_state                   # door 2 state {0,1} = {closed,open}
    x06 = goal_key                      # goal location {0,1,2} = {[5,1],[6,3],[5,6]}
    x07 = key_key                       # key location {0,1,2} = {[1,1],[2,3],[1,6]}
    x08 = 0                             # key state {0,1} = {in env, obtained}

    print('Goal Location:')
    print(goal_loc[goal_key])
    print('Key Location:')
    print(key_loc[key_key])

    # Define initial state
    x0 = (x01,x02,x03,x04,x05,x06,x07,x08)
    print('Initial State:')
    print(x0)

    # TIME HORIZON
    T = 200 # some arbitrarily large amount

    # STATE SPACE
    X = np.zeros((h,h,4,2,2,3,3,2))

    # VALUE FUNCTION
    V = np.ones((h,h,4,2,2,3,3,2,T))*np.inf   

    # OPTIMAL CONTROL POLICY
    optim_act_seq = {}

    # CONTROL SPACE - ('ACTION',COST)
    U = dict([(MF,3),(TL,3),(TR,3),(PK,1),(UD,1)])
    
    # TERMINAL COST    
    q = terminalCost(X, goal_loc)
    
    # Static wall locations for all environments
    walls = [[4,1],[4,3],[4,4],[4,6]]

    # Terminal Time Value Function
    V[:,:,:,:,:,:,:,:,T-1] = q
    Q = np.zeros(5)

    for t in range(T-1,0,-1):  # loop over time

        # Loop over X
        for x in np.ndindex(np.shape(X)):

            # Loop over U
            for u in range(0,5):

                # Compute x_prime (x_t+1)
                x_p = motionModel_B(x,u,info,walls,key_loc,goal_loc)

                # Define Cost Function for each different action
                Q[u] = stageCost_B(x,u,info,U,walls,key_loc,goal_loc) + V[x_p[0],x_p[1],x_p[2],x_p[3],x_p[4],x_p[5],x_p[6],x_p[7],t]

            # Define Value Function for each state x at time t
            # Only update a state's value function if less than past value function
            V[x][t-1] = min(Q) if V[x][t] > min(Q) else V[x][t]
            
        # When all value functions are the same as prior iteration, terminate loop
        if np.array_equal(V[:,:,:,:,:,:,:,:,t-1],V[:,:,:,:,:,:,:,:,t]) and t != T-1:
            print('All Values Converged at time:')
            print(t)
            break

    # Generating Optimal Policy starting from x0
    optim_act_seq = []
    seq_Value = []
    curState = x0
    while V[curState][t-1] > 0: # look at values starting at terminal time
        
        # check each control action starting at initial condition and return one that provides minimum cost
        
        seq_Value.append(V[curState][t-1])

        for u in range(0,5):
            x_cand = motionModel_B(curState,u,info,walls,key_loc,goal_loc)
            candidate_State = (x_cand[0],x_cand[1],x_cand[2],x_cand[3],x_cand[4],x_cand[5],x_cand[6],x_cand[7])
            Q[u] = V[candidate_State][t-1] # compute cost at all future states with control input         
        opt_act = np.argmin(Q)
        optim_act_seq.append(opt_act) # add optimal action to sequence       
        A = motionModel_B(curState,opt_act,info,walls,key_loc,goal_loc) # move to next state using the optimal action
        curState = (A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7])

    seq_Value.append(V[curState][t-1])        
    
    policy = []
    # Translate sequence to strings
    for n in range(0,len(optim_act_seq)):
        # policy for initial state
        policy.append(policyTable[optim_act_seq[n]])

    return policy, seq_Value

def partA():
    print('BEGINNING PART A')
    '''UNCOMMENT EACH PATH INDIVIDUALLY TO CHECK CODE'''
    env_path = './envs/doorkey-5x5-normal.env' # works
    #env_path = './envs/doorkey-6x6-normal.env' # works
    #env_path = './envs/doorkey-8x8-normal.env' # works
    #env_path = './envs/doorkey-6x6-direct.env'  # works
    #env_path = './envs/doorkey-8x8-direct.env'  # works
    #env_path = './envs/doorkey-6x6-shortcut.env' # works
    #env_path = './envs/doorkey-8x8-shortcut.env' # works
    print('Environment:')
    print(env_path)
    envNum = envDict[env_path] # used to determine which wall set to use
    env, info = load_env(env_path) # load an environment
    seq, seq_Val = doorkey_problem_A(envNum, info) # find the optimal action sequence
    seqStr = []
    for i in range(0,len(seq)):    
        seqStr.append(policyStrTable[seq[i]])
    print('Optimal Policy:')
    print(seqStr)
    print(seq_Val)
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save
    
def partB():
    print('BEGINNING PART B')
    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)
    print('Environment:')
    print(env_path)
    seq, seq_Val = doorkey_problem_B(env, info)
    seqStr = []
    for i in range(0,len(seq)):    
        seqStr.append(policyStrTable[seq[i]])        
    print('Optimal Policy:')
    print(seqStr)
    print(seq_Val)
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save

if __name__ == '__main__':
    '''Run doorkey.py in terminal to run scripts'''
    #partA() # see script above, must uncomment the environment strings to check different environments
    partB() # runs quite fast, but commenting out to check part A may be preferable    
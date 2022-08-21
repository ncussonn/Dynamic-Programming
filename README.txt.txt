ECE 276B Project 2 - Dynamic Programming

To run script, run doorkey.py in the terminal from
its original folder (dependencies on other files will be
broken otherwise).

Custom Functions:

+++++++++++++
  utils.py
+++++++++++++

Description:
Contains the functions necessary to run doorkey.py successfully. Must be
kept in same folder as doorkey.py if defaults are kept.

Functions:
__________
terminalCost(x,goal_coord)

Inputs: X, goal_coord
Outputs: q

Takes the state space X and goal coordinates for part A or B
and returns the terminal cost q for each state in the state space
at the time horizon T.

invalidAction_A(x,u,info,walls)

Inputs: x,u,info,walls
Outputs: True/False

For part A, takes a state x, control action u, information about the 
current environment (info) and wall coordinates as input.Returns boolean 
True if combination of x and u are invalid, False otherwise.

invalidAction_B(x,u,info,walls,key_loc,goal_loc)

Inputs: x,u,info,walls,key_loc,goal_loc
Outputs: True/False

For part B, takes a state x, control action u, information about the
current environment (info), wall coordinates, and key locations, key_loc, 
and lastly the goal locations, goal_loc. Returns boolean True if
combination of x and u are invalid, False otherwise.

stageCost_A(x,u,info,U,walls)

inputs: x,u,info,U,walls
outputs: l

For part A, takes state x, control action u, info from env, control action dict U,
and walls list for respective environment A. Returns the stage cost
of taking an action at state x as variable l. If a combination is
invalid, returns infinity, otherwises returns associated cost of u.

stageCost_B(x,u,info,U,walls,key_loc,goal_loc)

inputs: x,u,info,U,walls,key_loc,goal_loc
outputs: l

Takes state x, control action u, info from env, control action dict U,
wall coordinates list for B, key locations, and goal locations. Returns 
the stage cost of taking an action at state x as variable l. If a combination is
invalid, returns infinity, otherwises returns associated cost of u.

motionModel_A(x,u,info,walls)

inputs: x,u,info,walls
outputs: x_p

For part A, uses control action u on state of x, and updates the state if the 
motion is valid. Returns the altered, or unaltered state x prime (x_p)

motionModel_B(x,u,info,walls,key_loc,goal_loc)

inputs: x,u,info,walls,key_loc,goal_loc
outputs: x_p

For part B, uses control action u on state of x, and updates the state if the 
motion is valid. Returns the altered, or unaltered state x prime (x_p)


+++++++++++++
  doorkey.py
+++++++++++++

Description:
Main file, running this in the terminal will conduct Parts A and B of the
project. Contains two custom functions that complete part A and B respectfully.

Functions:
__________

doorkey_problem_A(env,info)

inputs: env, info
outputs: policy, seq_Value

Takes in environment class and environment information dictionary for part A
environment. Performs dynamic programming to return the optimal policy and 
associated value function sequence for the provided env.


doorkey_problem_B(env,info)

inputs: env, info
outputs: policy, seq_Value

Takes in environment class and environment information dictionary for part B
random environment. Performs dynamic programming to return the single
optimal policy for a given random initial state and its associated value function
sequence for the provided env.

##################
## Part A Notes ##
##################

To test different environments user must edit file by 
leaving only one desired uncommented file string name in the 
partA() function.


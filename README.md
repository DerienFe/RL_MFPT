#Code by Tiejun Wei; Edina Rosta; Simian Xing;
#2023 June

this repoo contains a serie of python/matlab scripts, constructing simple 1D/2D systems.
manipulating the FES, adding bias and calculating the mfpts.
the general goal is to find optim way to add the bias(or set of bias) to minimize the mfpt of 
the system from starting state to the end state.

In the RL folders, a simple Q-learning agent is constructed to interact with the environment. 
the reward function is defined as the mfpt of the system. 
the state is defined as the current position of the system. 
the action is defined as the bias added to the system.
the Q-table is updated by the Q-learning algorithm.

Unfinished work.
Finished work in folder: "1D_model_potential_Edina"; "2D_model_potential";

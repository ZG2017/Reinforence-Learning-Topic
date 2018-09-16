
# coding: utf-8

# Q learning 
import numpy as np
import pandas as pd
import time
import os
import math
import win_unicode_console

win_unicode_console.enable()



# In[199]:

class oneD_maze_env():
    def __init__(self,size_input):
        self.size = size_input
        self.location = 0

    def reset(self):
        self.location = 0
        
    def render(self):
        self.maze = np.empty((1,self.size),dtype = str)
        self.maze[0,:] = "-"
        self.maze[0,self.location] = "o"
        if self.maze[0,self.size-1] == "-":
            self.maze[0,self.size-1] = "T"
        df = pd.DataFrame(self.maze)
        print(df)
        
    def feedback(self,action,Q_table):
        done = False
        if action == "left" and self.location > 0:
            self.location-=1
        elif action == "right" and self.location < self.size-1:
            self.location+=1
        r = 0
        if self.location == self.size-1:
            done = True
            r = 1

        return self.location, r, done

def Q_learning(s,s_,a,r,Q_table,lr,gama,done):
    if not done:
        q_predict = r + gama*Q_table.loc[[s_],:].max(axis = 1).values[0]
    else:
        q_predict = r
    q_real = Q_table.loc[[s],:].max(axis = 1).values[0]
    Q_table.loc[[s],a] += lr * (q_predict - q_real)
    return Q_table
        
    
def main():
    lr = 0.01
    epsilon = 0.9
    reflash_rate = 0.3
    maze_size = 10
    gama = 0.9
    max_epoch = 300
    max_step = 100
    action = ["left","right"]
    #np.random.seed(200)
    
    # generate Q_table
    tmp = []
    for i in range(maze_size):
        tmp.append(i)
    Q_table = np.zeros((maze_size,len(action)))
    Q_table = pd.DataFrame(Q_table,index = tmp,columns = ["left","right"])
    env = oneD_maze_env(maze_size)
    for i in range(max_epoch):
        env.reset()
        s = env.location
        r = 0
        done = False
        os.system("cls")
        print(Q_table)
        print("%d-th epoch(es)"%i)
        env.render()
        time.sleep(reflash_rate)
        for j in range(max_step):
            if done:
                break

            if np.random.random() > epsilon or Q_table.loc[s,:].all() == 0:
                a = action[np.random.randint(2)]
            else:
                a = Q_table.loc[[s],:].idxmax(axis = 1).values[0]
                
            s_, r, done = env.feedback(a ,Q_table)
            Q_learning(s,s_,a,r,Q_table,lr,gama,done)
            s = s_
            os.system("cls")
            print("%d-th epoch(es)"%i)
            env.render()
            time.sleep(reflash_rate)
            
        print("epoch finish!\n---------------------------------------\n")
        print(r)
        print(Q_table)
        
    print(Q_table)


# In[200]:

if __name__ == "__main__":
    main()


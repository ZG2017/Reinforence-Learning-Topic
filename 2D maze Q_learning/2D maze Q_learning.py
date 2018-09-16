
# coding: utf-8

# In[180]:

# Q learning 
import numpy as np
import pandas as pd
import time
import os
import math
import win_unicode_console

win_unicode_console.enable()

# In[199]:

class maze_env():
    def __init__(self,size_input):
        self.size = size_input
        self.location = [0,0]

    def reset(self):
        self.location = [0,0]
        
    def render(self):
        self.maze = np.empty((self.size,self.size),dtype = str)
        self.maze[:,:] = "-"
        self.maze[self.location[0],self.location[1]] = "o"
        if self.maze[self.size-2, self.size-2] == "-":
            self.maze[self.size-2, self.size-2] = "T"
        if self.maze[self.size-3, self.size-2] == "-":
            self.maze[self.size-3, self.size-2] = "x"
        if self.maze[self.size-2, self.size-3] == "-":
            self.maze[self.size-2, self.size-3] = "x"
        df = pd.DataFrame(self.maze)
        print(df)
        
    def feedback(self,action,Q_table):
        done = False
        if action == "up" and self.location[0] > 0:
            self.location[0]-=1
        elif action == "down" and self.location[0] < self.size-1:
            self.location[0]+=1
        elif action == "left" and self.location[1] > 0:
            self.location[1]-=1
        elif action == "right" and self.location[1] < self.size-1:
            self.location[1]+=1

        if self.location == [self.size-2,self.size-2]:
            r = 10
            done = True
        elif self.location == [self.size-3,self.size-2] or self.location == [self.size-2,self.size-3]:
            r = -10
            done = True
        else:
            r = 0
        
        return self.location, r, done

def Q_learning(s,s_,a,r,Q_table,lr,gama,done):
    if not done:
        q_predict = r + gama*Q_table.loc[[tuple(s_)],:].max(axis = 1).values[0]
    else:
        q_predict = r

    q_real = Q_table.loc[[tuple(s)],:].max(axis = 1).values[0]
    Q_table.loc[[tuple(s)],a] += lr * (q_predict - q_real)
    return Q_table
        
    
def main():
    lr = 0.01
    epsilon = 0.9
    reflash_rate = 0.0
    maze_size = 5
    gama = 0.9
    max_epoch = 150
    max_step = 100
    action = ["up","down","left","right"]
    #np.random.seed(200)
    
    # generate Q_table
    tmp = []
    for i in range(maze_size):
        for j in range(maze_size):
            tmp.append((i,j))
    Q_table = np.zeros((maze_size**2,4))
    Q_table = pd.DataFrame(Q_table,index = tmp,columns = ["up","down","left","right"])
    env = maze_env(maze_size)
    for i in range(max_epoch):
        env.reset()
        s = env.location.copy()
        r = 0
        done = False
        os.system("cls")
        print("%d-th epoch(es)"%i)
        env.render()
        time.sleep(reflash_rate)
        for j in range(max_step):
            if done:
                break
            if np.random.random() > epsilon or Q_table.loc[[tuple(s)],:].all().all() == 0:
                a = action[np.random.randint(4)]
            else:
                a = Q_table.loc[[tuple(s)],:].idxmax(axis = 1).values[0]

            s_, r, done = env.feedback(a ,Q_table)
            Q_learning(s,s_,a,r,Q_table,lr,gama,done)
            s = s_.copy()

            time.sleep(reflash_rate)
            os.system("cls")
            print("%d-th epoch(es)"%i)
            env.render()
        print("epoch finish!\n---------------------------------------")
        #print(r)
        #print(Q_table)
        if r == 10:
            print("successfully done！")
        elif r == -10:
            print("you\'re dead!")
        aaa = input("wait a mins!")
        #time.sleep(reflash_rate)
    print(Q_table)


# In[200]:

if __name__ == "__main__":
    main()


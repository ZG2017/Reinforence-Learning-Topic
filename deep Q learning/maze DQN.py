
# coding: utf-8

# In[180]:

# Q learning 
import numpy as np
import pandas as pd
import time
import os
import math
import win_unicode_console
import tensorflow as tf

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
        '''
        if self.maze[self.size-2, self.size-3] == "-":
            self.maze[self.size-2, self.size-3] = "x"
        '''
        df = pd.DataFrame(self.maze)
        print(df)
       
    def feedback(self,action):
        done = False
        r = 0
        if action == "up":
            if self.location[0] > 0:
                self.location[0]-=1
                r = 0
            else:
                r = -0.2 
        elif action == "down":
            if self.location[0] < self.size-1:
                self.location[0]+=1
                r = 0
            else:
                r = -0.2
        elif action == "left":
            if self.location[1] > 0:
                self.location[1]-=1
                r = 0
            else:
                r = -0.2
        elif action == "right":
            if self.location[1] < self.size-1:
                self.location[1]+=1
                r = 0
            else:
                r = -0.2

        if self.location == [self.size-2,self.size-2]:
            r = 10
            done = True
        elif self.location == [self.size-3,self.size-2]: #or self.location == [self.size-2,self.size-3]:
            r = -1
            done = True
        
        return self.location, r, done
'''
def Q_learning(s,s_,a,r,Q_table,lr,gama,done):
    if not done:
        q_predict = r + gama*Q_table.loc[[tuple(s_)],:].max(axis = 1).values[0]
    else:
        q_predict = r

    q_real = Q_table.loc[[tuple(s)],:].max(axis = 1).values[0]
    Q_table.loc[[tuple(s)],a] += lr * (q_predict - q_real)
    return Q_table
'''

class DQN():
    def __init__(self,lr,epsilon_max,gama,number_of_states,number_of_actions,\
                 replace_steps,memory_size,batch_size,action):   # s,s_,a,r,Q_table,lr,gama,done,
        self.lr = lr
        self.epsilon_init = 0.1 
        self.epsilon_max = epsilon_max
        self.epsilon_increase = (self.epsilon_max-self.epsilon_init)/2000
        self.gama = gama
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.replace_steps = replace_steps
        self.replace_counter = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size,self.number_of_states*2+2))  # s,s_,r,a
        self.memory_counter = 0
        self.loss_his = []
        self.action = action
        self.flag = False
        self.replace_his = []
        self.sess = tf.Session()

    
    def build_net(self):
        # eval net-------------------------------------------
        self.s = tf.placeholder(tf.float32,[None,self.number_of_states],name = "s")
        self.q_target = tf.placeholder(tf.float32,[None,self.number_of_actions],name = "a")
        with tf.variable_scope("eval_net"):
            c_names = ["eval_net_params",tf.GraphKeys.GLOBAL_VARIABLES]
            self.n_l1 = 30
            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1",[self.number_of_states,self.n_l1],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b1 = tf.get_variable("b1",[1,self.n_l1],initializer=tf.constant_initializer(0.1),collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)
            
            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2",[self.n_l1,self.number_of_actions],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b2 = tf.get_variable("b2",[1,self.number_of_actions],initializer=tf.constant_initializer(0.1),collections=c_names)
                self.q_eval = tf.matmul(l1,w2)+b2

            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                
        # target net-----------------------------------------
        self.s_ = tf.placeholder(tf.float32,[None,self.number_of_states],name = "s_")
        with tf.variable_scope("target_net"):
            c_names = ["target_net_params",tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 30
            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1",[self.number_of_states,n_l1],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b1 = tf.get_variable("b1",[1,n_l1],initializer=tf.constant_initializer(0.1),collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_,w1)+b1)
            
            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2",[n_l1,self.number_of_actions],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b2 = tf.get_variable("b2",[1,self.number_of_actions],initializer=tf.constant_initializer(0.1),collections=c_names)
                self.q_next = tf.matmul(l1,w2)+b2

        self.sess.run(tf.global_variables_initializer())

    # save memoery
    def save_memory(self,s,s_,a,r):
        tmp_s = s[0]*int(math.sqrt(self.number_of_states))+s[1]
        tmp_s_ = s_[0]*int(math.sqrt(self.number_of_states))+s_[1]
        tmp_a = self.action.index(a)
        self.memory[self.memory_counter,tmp_s] = 1
        self.memory[self.memory_counter,self.number_of_states+tmp_s_] = 1
        self.memory[self.memory_counter,2*self.number_of_states] = tmp_a
        self.memory[self.memory_counter,2*self.number_of_states+1] = r
        self.memory_counter += 1
        if self.memory_counter == self.memory_size:
            self.flag = True
        self.memory_counter = self.memory_counter%self.memory_size

    def choice_action(self,s):
        tmp_s = np.zeros((1,self.number_of_states))
        tmp = s[0]*int(math.sqrt(self.number_of_states))+s[1]
        tmp_s[0,tmp] = 1
        action_base = self.sess.run(self.q_eval,feed_dict={self.s:tmp_s})
        tmp_a = np.argmax(action_base)

        if np.random.random() > self.epsilon_init:
            a = self.action[np.random.randint(4)]
        else:
            a = self.action[tmp_a]
        return a

    def replace(self):
        t_params = tf.get_collection("target_net_params")
        e_params = tf.get_collection("eval_net_params")
        self.sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])


    def learning(self):
        
        # see if needed to update target net
        if self.replace_counter%self.replace_steps == 0:
            self.replace()
            print("Target net has been replaced!")
            self.replace_his.append(0.1)
        else:
            self.replace_his.append(0)
        self.replace_counter += 1
        

        if self.flag == False:
            tmp_high = self.memory_counter
        else:
            tmp_high = self.memory_size
        batch_number = np.random.randint(low = 0,high = tmp_high,size = self.batch_size)
        batch_memory = self.memory[batch_number,:]

        q_eval,q_next = self.sess.run([self.q_eval,self.q_next],
        feed_dict = {self.s:batch_memory[:,:self.number_of_states],\
                     self.s_:batch_memory[:,self.number_of_states:2*self.number_of_states]})
        
        q_target = q_eval.copy()
        q_target[np.array(range(self.batch_size)),batch_memory[:,2*self.number_of_states].astype(int)] = \
        np.max(q_next,axis = 1)*self.gama + batch_memory[:,2*self.number_of_states+1]
        
        _,loss = self.sess.run([self.optimizer,self.loss],feed_dict={self.s:batch_memory[:,:self.number_of_states],
                                                                     self.q_target:q_target})

        self.loss_his.append(loss)

        if self.epsilon_init<self.epsilon_max:
            self.epsilon_init += self.epsilon_increase

    def plot(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        tmp = np.array(range(len(self.loss_his)))
        plt.plot(tmp,self.loss_his,"b-",tmp,self.replace_his,"r.")  
        plt.show()


def main():
    lr = 0.001
    epsilon = 0.9
    reflash_rate = 0.0
    maze_size = 4
    gama = 0.9
    max_epoch = 400
    max_step = 50
    action = ["up","down","left","right"]
    #np.random.seed(200)
    
    env = maze_env(maze_size)
    DQN_mou = DQN(lr = lr,
                    epsilon_max = epsilon,
                    gama = gama,
                    number_of_states = maze_size**2,
                    number_of_actions= 4,
                    replace_steps = 200,
                    memory_size = 2000,
                    batch_size = 200,
                    action = action)
    DQN_mou.build_net()
    
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
                print("number of step(s): %d"%(j))
                break
            a = DQN_mou.choice_action(s)
            s_, r, done = env.feedback(a)
            DQN_mou.save_memory(s,s_,a,r)
            DQN_mou.learning()
            s = s_.copy()

            os.system("cls")
            print("%d-th epoch(es)"%i)
            env.render()
            time.sleep(reflash_rate)
        print("epoch finish!\n---------------------------------------")
        time.sleep(0.3)
    DQN_mou.plot()
    print("All done!")


# In[200]:

if __name__ == "__main__":
    main()


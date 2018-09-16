import gym
import numpy as np
import time
import os
import math
import win_unicode_console
import tensorflow as tf
import win_unicode_console

win_unicode_console.enable()

env = gym.make('CartPole-v0')  
env = env.unwrapped

class DQN():
    def __init__(self,lr,epsilon_max,gamma,number_of_states,number_of_actions,\
                 replace_steps,memory_size,batch_size):  
        self.lr = lr
        self.epsilon_init = 0.0
        self.epsilon_max = epsilon_max
        self.epsilon_increase = 0.001
        self.gamma = gamma
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.replace_steps = replace_steps
        self.replace_counter = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size,self.number_of_states*2+2))  # s,s_,r,a
        self.memory_counter = 0
        self.loss_his = []
        self.replace_his = []
        self.sess = tf.Session()

    
    def build_net(self):
        # eval net-------------------------------------------
        self.s = tf.placeholder(tf.float32,[None,self.number_of_states],name = "s")
        self.q_target = tf.placeholder(tf.float32,[None,self.number_of_actions],name = "a")
        with tf.variable_scope("eval_net"):
            c_names = ["eval_net_params",tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 30
            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1",[self.number_of_states,n_l1],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b1 = tf.get_variable("b1",[1,n_l1],initializer=tf.constant_initializer(0.1),collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)
            
            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2",[n_l1,self.number_of_actions],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b2 = tf.get_variable("b2",[1,self.number_of_actions],initializer=tf.constant_initializer(0.1),collections=c_names)
                self.q_eval = tf.matmul(l1,w2)+b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
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
        tmp = self.memory_counter%self.memory_size
        self.memory[tmp,:self.number_of_states] = s
        self.memory[tmp,self.number_of_states:2*self.number_of_states] = s_
        self.memory[tmp,2*self.number_of_states] = a
        self.memory[tmp,2*self.number_of_states+1] = r
        self.memory_counter += 1

    def choose_action(self,s):
        action_base = self.sess.run(self.q_eval,feed_dict={self.s:s})
        tmp_a = np.argmax(action_base)

        if np.random.uniform() < self.epsilon_init:
            a = tmp_a
        else:
            a = np.random.randint(self.number_of_actions)
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
        
        if self.memory_counter > self.memory_size:
            batch_number = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_number = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[batch_number,:]

        q_eval,q_next = self.sess.run([self.q_eval,self.q_next],
        feed_dict = {self.s:batch_memory[:,:self.number_of_states],\
                     self.s_:batch_memory[:,self.number_of_states:2*self.number_of_states]})
        
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.number_of_states*2].astype(int)
        reward = batch_memory[:, self.number_of_states*2 + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        '''
        q_target[np.array(range(self.batch_size)),batch_memory[:,2*self.number_of_states].astype(int)] = \
        np.max(q_next,axis = 1)*self.gama + batch_memory[:,2*self.number_of_states+1]
        '''

        _,loss = self.sess.run([self.optimizer,self.loss],feed_dict={self.s:batch_memory[:,:self.number_of_states],
                                                                     self.q_target:q_target})

        self.loss_his.append(loss)

        if self.epsilon_init<self.epsilon_max:
            self.epsilon_init += self.epsilon_increase

    def plot(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        tmp = np.array(range(len(self.loss_his)))
        plt.plot(tmp,self.loss_his,"b-")  #,tmp,self.replace_his,"r."
        plt.show()


lr = 0.01
epsilon = 0.9
gamma = 0.9
max_epoch = 100

DQN_mou = DQN(lr = lr,
              epsilon_max = epsilon,
              gamma = gamma,
              number_of_states = env.observation_space.shape[0],
              number_of_actions= env.action_space.n,
              replace_steps = 100,
              memory_size = 2000,
              batch_size = 32)

DQN_mou.build_net()

total_steps = 0
for i_episode in range(max_epoch):

    observation = env.reset()
    observation = np.expand_dims(observation,axis = 0)
    ep_r = 0
    while True:
        env.render() 

        action = DQN_mou.choose_action(observation) 

        observation_, reward, done, info = env.step(action)
        x, x_dot, theta, theta_dot = observation_   # 细分开, 为了修改原配的 reward
        observation_ = np.expand_dims(observation_,axis = 0)

        # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2   # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率

        DQN_mou.save_memory(observation, observation_, action, reward)

        if total_steps > 1000:
            DQN_mou.learning() 

        ep_r += reward
        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(DQN_mou.epsilon_init, 2))
            break

        observation = observation_
        total_steps += 1
DQN_mou.plot()


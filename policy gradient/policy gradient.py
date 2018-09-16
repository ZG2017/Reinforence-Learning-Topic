import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')   
env = env.unwrapped 

class policy_gradient():
    def __init__(self,number_of_states,number_of_actions,lr,gamma):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.lr = lr
        self.gamma = gamma
        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []
        self.loss_his = []
        self.sess = tf.Session()
        

    def build_net(self):
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num") 
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        self.s = tf.placeholder(tf.float32,[None,self.number_of_states],name = "s")
        with tf.variable_scope("eval_net"):
            n_l1 = 30
            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1",[self.number_of_states,n_l1],initializer=tf.random_normal_initializer(0,0.3))
                b1 = tf.get_variable("b1",[1,n_l1],initializer=tf.constant_initializer(0.1))
                l1 = tf.nn.tanh(tf.matmul(self.s,w1)+b1)
            
            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2",[n_l1,self.number_of_actions],initializer=tf.random_normal_initializer(0,0.3))
                b2 = tf.get_variable("b2",[1,self.number_of_actions],initializer=tf.constant_initializer(0.1))
                self.q_eval = tf.matmul(l1,w2)+b2
        
        self.pro = tf.nn.softmax(self.q_eval)

        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.q_eval, labels=self.tf_acts)
        self.loss = tf.reduce_mean(neg_log_prob*self.tf_vt)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self,s):
            s = np.expand_dims(s,axis = 0)
            pro_weight = self.sess.run(self.pro,feed_dict={self.s:s})
            a = np.random.choice(range(self.number_of_actions),p = pro_weight[0])
            return a

    def save_memory(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learning(self):
        discounted_ep_rs_norm = self.discount_reward()
        _,tmp_loss = self.sess.run([self.optimizer,self.loss],feed_dict = {self.s:np.vstack(self.ep_obs),
                                                                           self.tf_acts:np.array(self.ep_as),
                                                                           self.tf_vt:discounted_ep_rs_norm})
        self.loss_his.append(tmp_loss)
        self.ep_obs,self.ep_as,self.ep_rs = [],[],[]
        return discounted_ep_rs_norm
        
    def discount_reward(self):
        discount_rd = np.zeros_like(self.ep_rs)
        tmp = 0
        for i in reversed(range(len(self.ep_rs))):
            tmp += tmp*self.gamma + self.ep_rs[i]
            discount_rd[i] = tmp
        
        # normalize
        discount_rd -= np.mean(discount_rd)
        discount_rd /= np.std(discount_rd)
        return discount_rd

    def plot(self):
        fig = plt.figure()
        tmp = np.array(range(len(self.loss_his)))
        plt.plot(tmp,self.loss_his,"b-")  #,tmp,self.replace_his,"r."
        plt.show()


lr = 0.008
gamma = 0.90
max_epoch = 600
RL = policy_gradient(number_of_states =  env.observation_space.shape[0],
                     number_of_actions =  env.action_space.n,
                     lr = lr,
                     gamma = gamma)
RL.build_net()
for i_episode in range(max_epoch):
    observation = env.reset()
    #observation = np.expand_dims(observation,axis = 0)

    while True:
        env.render()

        tmp_reward = np.sum(RL.ep_rs)

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.save_memory(observation, action, reward)

        if done:   
            print("episode:", i_episode, "  reward:", tmp_reward)

            vt = RL.learning() 
            
            if i_episode == 0:
                plt.plot(sum)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
        observation = observation_

RL.plot()
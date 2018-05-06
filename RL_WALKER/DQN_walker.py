# DSA final project Walker training
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
from RL_WALKER.utils import Memory


# seeds for randomlization
np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 2000
LR_A = 0.0005  # learning rate
LR_C = 0.0005  # learning rate
GAMMA = 0.999
REPLACE_ITER_A = 1700
REPLACE_ITER_C = 1500
MEMORY_CAPACITY = 200000
BATCH_SIZE = 32
DISPLAY_THRESHOLD = 100  # display until the running reward > 100
DATA_PATH = './data' # path for save data file
LOAD_MODEL = False
SAVE_MODEL_ITER = 10000  # Model save iteration
RENDER = False # show render or not
OUTPUT_GRAPH = False
ENV_NAME = 'BipedalWalker-v2' # Gym environment

GLOBAL_STEP = tf.Variable(0, trainable=False)
INCREASE_GS = GLOBAL_STEP.assign(tf.add(GLOBAL_STEP, 1))
LR_A = tf.train.exponential_decay(LR_A, GLOBAL_STEP, 10000, .97, staircase=True)
LR_C = tf.train.exponential_decay(LR_C, GLOBAL_STEP, 10000, .97, staircase=True)
END_POINT = (200 - 10) * (14/30)    # from game

env = gym.make(ENV_NAME)
env.seed(1)

STATE_DIM = env.observation_space.shape[0]  # 24
ACTION_DIM = env.action_space.shape[0]  # 4
ACTION_BOUND = env.action_space.high    # [1, 1, 1, 1]

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')



# STRUCTURE FOR DQN
class Choose_action(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.01)
            init_b = tf.constant_initializer(0.01)
            net = tf.layers.dense(s, 500, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2', trainable=trainable)

            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    # gradient calculation
    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads_and_vars = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads_and_vars, self.e_params), global_step=GLOBAL_STEP)


class Q_regression(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('abs_TD'):
            self.abs_td = tf.abs(self.target_q - self.q)
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=GLOBAL_STEP)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.01)
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 700
                # combine the action and states together in this way
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            with tf.variable_scope('l2'):
                net = tf.layers.dense(net, 20, activation=tf.nn.relu, kernel_initializer=init_w,
                                      bias_initializer=init_b, name='l2', trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_, ISW):
        _, abs_td = self.sess.run([self.train_op, self.abs_td], feed_dict={S: s, self.a: a, R: r, S_: s_, self.ISWeights: ISW})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1
        return abs_td


if __name__ == "__main__":
    sess = tf.Session()

    # Create the DQN model
    actor = Choose_action(sess, ACTION_DIM, ACTION_BOUND, LR_A, REPLACE_ITER_A)
    critic = Q_regression(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    M = Memory(MEMORY_CAPACITY)

    saver = tf.train.Saver(max_to_keep=100)

    if LOAD_MODEL:
        all_ckpt = tf.train.get_checkpoint_state('./data', 'checkpoint').all_model_checkpoint_paths
        saver.restore(sess, all_ckpt[-1])
    else:
        if os.path.isdir(DATA_PATH):
            shutil.rmtree(DATA_PATH)
        os.mkdir(DATA_PATH)
        sess.run(tf.global_variables_initializer())


    # writing tensorboard
    if OUTPUT_GRAPH:
        tf.summary.FileWriter('logs', graph=sess.graph)

    # hyperparameters for control exploration
    var = 3
    var_min = 0.01

    for i_episode in range(MAX_EPISODES):
        # state vector hull angle speed, angular velocity, horizontal speed,
        # vertical speed, position of joints and joints
        # angular speed, legs contact with ground, and 10 lidar rangefinder measurements.
        s = env.reset()
        ep_r = 0
        # total_action = np.zeros((1, 400, 600, 3))
        total_action = []

        while True:
            # for visualization
            if RENDER:
                frame = env.render("rgb_array")
                total_action.append(frame)
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), -1, 1)  # add randomness to action selection for exploration
            s_, r, done, _ = env.step(a)  # r = total 300+ points up to the far end. If the robot falls, it gets -100.

            # -100 is too large for network to tuning, we manually set it to -2
            if r == -100:
                r = -2

            ep_r += r

            transition = np.hstack((s, a, [r], s_))
            max_p = np.max(M.tree.tree[-M.tree.capacity:])
            M.store(max_p, transition)

            if GLOBAL_STEP.eval(sess) > MEMORY_CAPACITY / 20:
                var = max([var * 0.9999, var_min])  # decay the action randomness
                tree_idx, b_M, ISWeights = M.prio_sample(BATCH_SIZE)  # for critic update
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                abs_td = critic.learn(b_s, b_a, b_r, b_s_, ISWeights)
                actor.learn(b_s)
                for i in range(len(tree_idx)):  # update priority
                    idx = tree_idx[i]
                    M.update(idx, abs_td[i])

            if GLOBAL_STEP.eval(sess) % SAVE_MODEL_ITER == 0:
                ckpt_path = os.path.join(DATA_PATH, 'DDPG.ckpt')
                save_path = saver.save(sess, ckpt_path, global_step=GLOBAL_STEP, write_meta_graph=False)
                print("\nSave Model %s\n" % save_path)

            if done:
                if "running_r" not in globals():
                    running_r = ep_r
                else:
                    running_r = 0.95 * running_r + 0.05 * ep_r
                if running_r > DISPLAY_THRESHOLD:
                    RENDER = True
                else:
                    RENDER = False

                done = '| Achieve ' if env.unwrapped.hull.position[0] >= END_POINT else '| -----'

                # render to save the visualization file
                if RENDER:
                    np.save("./action/action_{}.npy".format(sess.run(GLOBAL_STEP)), np.array(total_action))

                break

            s = s_
            # update parameters
            sess.run(INCREASE_GS)

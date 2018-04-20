import gym
import cv2
import numpy as np
from PIL import Image
# environment generating
ENV_NAME = 'BipedalWalker-v2'
env = gym.make(ENV_NAME)
# env.seed(1)

# start render for visualization
env.render()
env.reset()
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output_success.avi', fourcc, 60.0, (600, 400))


actions = np.load(r"D:\Reinforcement-Learning-for-DSA\RL_WALKER\action\action_415593.npy")
print(actions.shape)
for i in actions:
    i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
    out.write(i)
    # cv2.imshow('frame', i)
    # cv2.waitKey()

out.release()
cv2.destroyAllWindows()
from util import *
from env import *
from gym.wrappers import TimeLimit
#register env.
from gym.envs.registration import register
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})


register(
    id='pos_1D-v0',
    entry_point='env:pos_1D_env',
    max_episode_steps=500, #how many propagation steps we allow.
    reward_threshold=0.0,
    kwargs={'N': 100,
            'kT': 0.5981,
            'state_start': 8,
            'state_end': 89,
            'num_gaussians': 10,
            'time_step': 0.1,
            'simulation_steps': 50000,
            'render_mode': None,
            }
)

env = gym.make('pos_1D-v0')
#envs = gym.vector.make('pos_1D-v0', num_envs=2)
#env = gym.wrappers.RecordEpisodeStatistics(env)
#env = gym.wrappers.RecordVideo(env, f"videos/{'test1'}")
env.reset()
env.render()
#test if the environment is working
#we put zero bias several times, and see if the system stays at local minimum.
test_bias = np.zeros(100)
#we add 2 gaussians at position 8.
"""test_bias[8] = 1
test_bias[9] = 1
test_bias[7] = 1
for i in range(10):
    observation, reward, done, info = env.step(test_bias)
    env.render()
    print("current position is: ", env.state)
"""

#we load the trained model and see if it works.

from ppo_trainer import *
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
envs = gym.vector.SyncVectorEnv([make_env(1 + i, i, 'test') for i in range(8)])
agent = Agent(envs).to(device)

agent.load_state_dict(torch.load("./RL_pos_PPO_1D/models/ppo_trainer__1__1690192924.pt"), strict=False)
test_pos = torch.Tensor(envs.reset()).to(device)
action, logprob, _, value = agent.get_action_and_value(test_pos)
print("all done")
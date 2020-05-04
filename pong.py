import torch
import gym
from gym import envs
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms
import sys



rendered = sys.argv[1]



lenobs = 21190
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(lenobs,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,3)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)

    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.normalize(y,dim=0)
        y = F.relu(self.l2(y))
        y = F.normalize(y,dim=0)
        actor = F.log_softmax(self.actor_lin1(y),dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic

env = gym.make('Pong-v0')
moveMapping = {
    0:0,
    1:2,
    2:3
}


def play(model, f):
    score = 0.0
    observation = env.reset()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
    ])
    
    model.eval()
    done = False
   

    while done == False:
        if f==1:
            env.render()
        pobservation = torch.from_numpy(observation).permute(2,0,1)
        cropped_image = transforms.functional.crop(preprocess(pobservation),32,15,163,130)
        gs_image = transforms.functional.to_grayscale(cropped_image)
        gs_tensor = transforms.ToTensor()(gs_image)
        flattened_pobservation = gs_tensor.view(-1).float()
        policy, value = model(flattened_pobservation)
        sampler = Categorical(policy)
        action = sampler.sample()
        observation, reward, done, log = env.step(moveMapping[action.item()])
        if f=='1':
            env.render()
        if reward == 1.0:
            score+=1.0

    if f=='1':
        env.close()
    
    return score




# Loading saved model
model = ActorCritic()
model_path = './a2c_seq.pth'
m1 = torch.load(model_path)
model.load_state_dict(m1)

if rendered == '1':
    print('Score obtained: {}'.format(play(model,1)))
    print()

else:

    score_distribution = [0 for _ in range(22)]
    nog = int(input('Enter Number of games\n'))  #number of games
    for _ in range(nog):
        score_distribution[int(play(model,0))]+=1



    print('Score        ',end='')
    for i in range(22):
        print(i, end='\t')

    print()
    print('Frequency    ',end='')
    for i in score_distribution:
        print(i,end='\t')

    print()
import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn

from modules import PolicyNet, ValueNet
from simple_network import SimpleNet
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None

class TrainDaggerBC:

    def __init__(self, env, model, expert_model, optimizer, states, actions, device="cpu", mode="DAgger"):
        """
        Initializes the TrainDAgger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_model: the expert model that provides the expert actions.
            device: the device to be used for training.
            mode: the mode to be used for training. Either "DAgger" or "BC".

        """
        self.env = env
        self.model = model
        self.expert_model = expert_model
        self.optimizer = optimizer
        self.device = device
        model.set_device(self.device)
        self.expert_model = self.expert_model.to(self.device)

        self.mode = mode

        if self.mode == "BC":
            self.states = []
            self.actions = []
            self.timesteps = []
            for trajectory in range(states.shape[0]):
                trajectory_mask = states[trajectory].sum(axis=1) != 0
                self.states.append(states[trajectory][trajectory_mask])
                self.actions.append(actions[trajectory][trajectory_mask])
                self.timesteps.append(np.arange(0, len(trajectory_mask)))
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.timesteps = np.concatenate(self.timesteps, axis=0)

            self.clip_sample_range = 1
            self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        else:
            self.states = None
            self.actions = None
            self.timesteps = None

    def generate_trajectory(self, env, policy):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
        """

        states, old_actions, timesteps, rewards = [], [], [], []

        done, trunc = False, False
        cur_state, _ = env.reset()  
        t = 0
        while (not done) and (not trunc):
            with torch.no_grad():
                p = policy(torch.from_numpy(cur_state).to(self.device).float().unsqueeze(0), torch.tensor(t).to(self.device).long().unsqueeze(0))
            a = p.cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(a)

            states.append(cur_state)
            old_actions.append(a)
            timesteps.append(t)
            rewards.append(reward)

            t += 1

            cur_state = next_state

        return states, old_actions, timesteps, rewards

    def call_expert_policy(self, state):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action = self.expert_model.choose_action(state_tensor, deterministic=True).cpu().numpy()
            action = np.clip(action, -1, 1)[0]
        return action

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        new_states = []
        new_actions = []
        new_timesteps = []
        for num in range(num_trajectories_per_batch_collection):
            states, old_actions, timesteps, rewards = self.generate_trajectory(self.env, self.model)
            for state in states:
                new_actions.append(self.call_expert_policy(state))
            new_states.extend(states)
            new_timesteps.extend(timesteps)
        if self.timesteps is not None:
            self.timesteps = np.concatenate((self.timesteps, np.array(new_timesteps)), axis = 0)
        else:
            self.timesteps = np.array(new_timesteps)

        if self.states is not None:
            self.states = np.concatenate((self.states, np.array(new_states)), axis = 0)
        else:
            self.states = np.array(new_states)

        if self.actions is not None:
            self.actions = np.concatenate((self.actions, np.array(new_actions)), axis = 0)
        else:
            self.actions = np.array(new_actions)

            
            




        # END STUDENT SOLUTION

        return rewards

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories. Use for behavior cloning.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.
        
        NOTE: you will need to call self.generate_trajectory in this function.
        """
        # BEGIN STUDENT SOLUTION
        rewards = []
        for _ in range(num_trajectories_per_batch_collection):
            rewards.append(sum(self.generate_trajectory(self.env, self.model)[-1]))
        # END STUDENT SOLUTION

        return rewards

    def train(
        self, 
        num_batch_collection_steps, 
        num_BC_training_steps=20000,
        num_training_steps_per_batch_collection=1000, 
        num_trajectories_per_batch_collection=20, 
        batch_size=64, 
        print_every=500, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        Train the model using BC or DAgger

        Args:
            num_batch_collection_steps: the number of times to collecta batch of trajectories from the current policy.
            num_BC_training_steps: the number of iterations to train the model using BC.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for BC, you will need to call the self.training_step function and self.generate_trajectories function.
        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        # BEGIN STUDENT SOLUTION
        losses = []
        mean = []
        median = []
        mx = []
        if self.mode == "BC" :
            for i in range(num_BC_training_steps):
                losses.append(self.training_step(batch_size))
                if i%num_training_steps_per_batch_collection == 0:
                    rewards = np.array(self.generate_trajectories(num_trajectories_per_batch_collection))
                    mean.append(rewards.mean())
                    median.append(np.median(rewards))
                    mx.append(rewards.max())
        else:
            for i in range(num_batch_collection_steps):
                self.update_training_data()
                for j in range(num_training_steps_per_batch_collection):
                    losses.append(self.training_step(batch_size))
                rewards = np.array(self.generate_trajectories(num_trajectories_per_batch_collection))
                mean.append(rewards.mean())
                median.append(np.median(rewards))
                mx.append(rewards.max())
                



        # END STUDENT SOLUTION

        return losses, mean, median, mx

    def training_step(self, batch_size):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps = self.get_training_batch(batch_size=batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)

        loss_fn = nn.MSELoss()
        self.optimizer.zero_grad()
        predicted_actions = self.model(states, timesteps)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[indices], device=self.device).float()
        actions = torch.tensor(self.actions[indices], device=self.device).float()
        timesteps = torch.tensor(self.timesteps[indices], device=self.device)
            
        
        return states, actions, timesteps

def run_training():
    """
    Simple Run Training Function

    """

    env = gym.make('BipedalWalker-v3') # , render_mode="rgb_array"

    model_name = "super_expert_PPO_model"
    expert_model = PolicyNet(24, 4)
    model_weights = torch.load(f"data/models/{model_name}.pt", map_location=torch.device('cpu'))
    expert_model.load_state_dict(model_weights["PolicyNet"])

    states_path = "/Users/vansh/Downloads/expert_collected_trajectories 2/states_BC.pkl"
    actions_path = "/Users/vansh/Downloads/expert_collected_trajectories 2/actions_BC.pkl"

    with open(states_path, "rb") as f:
        states = pickle.load(f)
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)

    # BEGIN STUDENT SOLUTION
    model = SimpleNet(env.observation_space.shape[0], env.action_space.shape[0], hidden_layer_dimension = 128, max_episode_length=1600)
    optimizer = torch.optim.AdamW(model.parameters(), lr =  0.0001, weight_decay=0.0001)
    mode = "DAgger"
    if mode == "BC":
        # Trainer = TrainDaggerBC(env, model, expert_model, optimizer, states, actions, device="cpu", mode="BC")
        # losses, mean, median, mx = Trainer.train(1)
        # plt.figure()
        # plt.plot(np.arange(1, len(losses)+1), losses)
        # plt.xlabel('No. of Iterations')
        # plt.ylabel('Training Loss')
        # plt.title('Training Loss VS No. of Iterations')
        # plt.grid()
        # plt.savefig('Training Loss.png')
        # plt.figure()
        # plt.plot(np.arange(1, len(mean)+1), mean,label= 'Mean Reward')
        # plt.plot(np.arange(1, len(median)+1), median,label= 'Median Reward')
        # plt.plot(np.arange(1, len(mx)+1), mx,label= 'Maximum Reward')
        # plt.xlabel('No. of Training Intervals')
        # plt.ylabel('Reward')
        # plt.xticks(np.arange(1, len(mx)+1))
        # plt.title('Variation of Reward Statistics During Training')
        # plt.grid()
        # plt.legend(title = 'Statistic')
        # plt.savefig('Statistics.png')
        # print(losses[-1])
        # torch.save(model, 'BC_model.pth')
        model = torch.load('BC_model.pth')
        model.eval()
        env = gym.make('BipedalWalker-v3', render_mode="rgb_array")
        while(True):
            tot_reward = 0
            frames= []
            terminated = False
            t = 0
            state, _ = env.reset()
            frame = env.render()
            frames.append(Image.fromarray(frame))
            while(not terminated):
                with torch.no_grad():
                    policy = model(torch.tensor(state).float().unsqueeze(0), torch.tensor(t).long().unsqueeze(0))
                    action = policy[0].numpy()
                next_state, reward, terminated, trun, _ = env.step(action)
                t += 1
                state = next_state
                tot_reward += reward
                frame = env.render()
                frames.append(Image.fromarray(frame))
            if tot_reward < 0:
                break
        frames[0].save('BC_failure.gif', save_all = True, append_images = frames[1:], duration = 50, loop = 0)
        plt.show() ## Final Loss for BC: 0.06219020485877991
    else:
        Trainer = TrainDaggerBC(env, model, expert_model, optimizer, states, actions, device="cpu", mode="DAgger")
        losses, mean, median, mx = Trainer.train(num_batch_collection_steps=20, batch_size=128, num_training_steps_per_batch_collection=1000)
        plt.figure()
        plt.plot(np.arange(1, len(losses)+1), losses)
        plt.xlabel('No. of Iterations')
        plt.ylabel('Training Loss')
        plt.title('Variation of Training Loss with No. of Iterations for DAgger')
        plt.grid()
        plt.savefig('Training Loss DAgger_2.png')
        plt.figure()
        plt.plot(np.arange(1, len(mean)+1), mean,label= 'Mean Reward')
        plt.plot(np.arange(1, len(median)+1), median,label= 'Median Reward')
        plt.plot(np.arange(1, len(mx)+1), mx,label= 'Maximum Reward')
        plt.xlabel('No. of Training Intervals')
        plt.ylabel('Reward')
        plt.xticks(np.arange(1, len(mx)+1))
        plt.title('Variation of Reward Statistics During Training for DAgger')
        plt.grid()
        plt.legend(title = 'Statistic')
        plt.savefig('Statistics DAgger_2.png')
        print(losses[-1])
        torch.save(model, 'DAgger_model_2.pth')
        # model = torch.load('DAgger_model.pth')
        # model.eval()
        # env = gym.make('BipedalWalker-v3', render_mode="rgb_array")
        # while(True):
        #     tot_reward = 0
        #     frames= []
        #     terminated = False
        #     t = 0
        #     state, _ = env.reset()
        #     frame = env.render()
        #     frames.append(Image.fromarray(frame))
        #     while(not terminated):
        #         with torch.no_grad():
        #             policy = model(torch.from_numpy(state).float().unsqueeze(0), torch.tensor(t).long().unsqueeze(0))
        #             action = policy[0].numpy()
        #         next_state, reward, terminated, trun, _ = env.step(action)
        #         t += 1
        #         tot_reward += reward
        #         state = next_state
        #         frame = env.render()
        #         frames.append(Image.fromarray(frame))
        #     if tot_reward > 260:
        #         break
        # frames[0].save('DAgger_success.gif', save_all = True, append_images = frames[1:], duration = 50, loop = 0)
        # plt.show() ## Final Loss for BC: 0.06219020485877991
        ## Final Loss for Diff: 0.0406





    # END STUDENT SOLUTION

if __name__ == "__main__":
    run_training()
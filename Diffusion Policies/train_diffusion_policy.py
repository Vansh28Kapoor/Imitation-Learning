import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn

from tqdm import tqdm
import pickle

from diffusion_policy_transformer import PolicyDiffusionTransformer
from PIL import Image
from diffusers import DDPMScheduler, DDIMScheduler
import time
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

class TrainDiffusionPolicy:

    def __init__(
        self,
        env,
        model, 
        optimizer, 
        states_array, 
        actions_array, 
        device="cpu", 
        num_train_diffusion_timesteps=30,
        max_trajectory_length=1600,
    ):
        """
        Initializes the TrainDiffusionPolicy class. Creates necessary data structures and normalizes states AND actions.

        Args:
            env (gym.Env): The environment that the model is trained on.
            model (PolicyDiffusionTransformer): the model to train
            optimizer (torch.optim.Optimizer): the optimizer to use for training the model
            states_array (np.ndarray): the states to train on
            actions_array (np.ndarray): the actions to train on
            device (str): the device to use for training
            num_train_diffusion_timesteps (int): the number of diffusion timesteps to use for training
        """
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.states = states_array
        self.actions = actions_array

        self.action_dimension = self.actions.shape[-1]
        self.state_dimension = self.states.shape[-1]

        # clip all actions to be between -1 and 1, as this is the range that the environment expects
        self.clip_sample_range = 1
        self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        self.trajectory_lengths = [sum(1 for s in self.states[i] if np.sum(s) != 0) for i in range(len(self.states))]
        self.max_trajectory_length = max_trajectory_length

        model.set_device(self.device)

        # normalize states and actions
        all_states = np.concatenate([self.states[i, 0:self.trajectory_lengths[i]] for i in range(len(self.states))], axis=0)
        all_actions = np.concatenate([self.actions[i, 0:self.trajectory_lengths[i]] for i in range(len(self.actions))], axis=0)

        self.states_mean = np.mean(all_states, axis=(0))
        self.states_std = np.std(all_states, axis=(0))
        self.states = (self.states - self.states_mean) / self.states_std

        self.actions_mean = np.mean(all_actions, axis=(0))
        self.actions_std = np.std(all_actions, axis=(0))
        self.actions = (self.actions - self.actions_mean) / self.actions_std

        self.num_train_diffusion_timesteps = num_train_diffusion_timesteps

        # training and inference schedulers for diffusion
        self.training_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log", # variance is different for inference, see paper https://arxiv.org/pdf/2301.10677
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler.alphas_cumprod = self.inference_scheduler.alphas_cumprod.to(self.device)


    def get_inference_timesteps(self):
        """
        gets the timesteps to use for inference
        """
        self.inference_scheduler.set_timesteps(self.num_train_diffusion_timesteps, device=self.device)
        return self.inference_scheduler.timesteps

    def diffusion_sample(
        self,
        previous_states, 
        previous_actions,
        episode_timesteps,
        previous_states_padding_mask=None,
        previous_actions_padding_mask=None,
        actions_padding_mask=None,
        max_action_len=3,
    ):
        """
        perform a single diffusion sample from noise to actions

        Args:
            previous_states (torch.Tensor): the previous states to condition on
            previous_actions (torch.Tensor): the previous actions to condition on
            episode_timesteps (torch.Tensor): the episode timesteps to condition on
            previous_states_padding_mask (torch.Tensor): the padding mask for the previous states
            previous_actions_padding_mask (torch.Tensor): the padding mask for the previous actions
            actions_padding_mask (torch.Tensor): the padding mask for the actions being predicted
            max_action_len (int): the maximum number of actions to predict

        NOTE: remember that you are predicting max_action_len actions, not just one
        """
        # BEGIN STUDENT SOLUTION
        timesteps = self.get_inference_timesteps()
        prev = torch.randn((1,max_action_len, self.action_dimension))
        for t in timesteps:
            # if t.item()>1:
            #     noise_levels = torch.randn((1,max_action_len, self.action_dimension))
            # else:
            #     noise_levels = torch.zeros((1,max_action_len, self.action_dimension))
            # noise_levels = torch.randn((1,max_action_len, self.action_dimension))
            # print(previous_states_padding_mask.shape,
            #             previous_actions_padding_mask.shape,
            #             )   
            predictions = self.model(previous_states,
                        previous_actions,
                        prev, 
                        episode_timesteps,
                        t.view(1,1),
                        previous_states_padding_mask,
                        previous_actions_padding_mask,
                        actions_padding_mask,
                        )
            current = self.inference_scheduler.step(model_output=predictions, timestep= t, sample = prev)
            prev = current[0]

        # END STUDENT SOLUTION
        return prev

    def sample_trajectory(
        self, 
        env, 
        num_actions_to_eval_in_a_row=3, 
        num_previous_states=5,
        num_previous_actions=4, 
    ):
        """
        run a trajectory using the trained model

        Args:
            env (gym.Env): the environment to run the trajectory in
            num_actions_to_eval_in_a_row (int): the number of actions to evaluate in a row
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on

        NOTE: use with torch.no_grad(): to speed up inference by not storing gradients
        NOTE: for the first few steps, make sure to add padding to previous states/actions - use False if a state/action should be included, and True if it should be padded
        NOTE: both states and actions should be normalized before being passed to the model, and the model outputs normalized actions that need to be denormalized
        NOTE: refer to the forward function of diffusion_policy_transformer to see how to pass in the inputs (tensor shapes, etc.)
        """
        # BEGIN STUDENT SOLUTION
        terminated = False
        truncated = False
        state, _ = env.reset()
        states_list = torch.tensor(state)
        states_list = states_list.unsqueeze(0)
        states_list = (states_list - self.states_mean) / self.states_std
        rewards = np.zeros(self.max_trajectory_length)
        t = 0
        actions_list = None
        while not terminated and not truncated:
            num_states = states_list.shape[0]
            if num_previous_states > num_states:
                previous_states = torch.vstack((states_list, torch.zeros((num_previous_states - num_states, self.state_dimension)))).float()
                if num_states == 1:
                    previous_actions = torch.zeros((num_previous_actions, self.action_dimension)).float()
                    previous_actions_padding_mask = torch.ones(num_previous_actions)
                else:
                    previous_actions = torch.vstack((actions_list, torch.zeros((num_previous_actions - num_states+1, self.action_dimension))))
                    previous_actions_padding_mask = torch.cat((torch.zeros(num_states-1), torch.ones(num_previous_states - num_states)))
                previous_states_padding_mask = torch.cat((torch.zeros(num_states), torch.ones(num_previous_states - num_states))).float()
                episode_timesteps = torch.arange(0,num_previous_states)
            else:
                previous_states = states_list
                previous_actions = actions_list
                previous_states_padding_mask = torch.zeros(num_states)
                previous_actions_padding_mask = torch.zeros(num_states-1)
                episode_timesteps = torch.arange(t-num_previous_states+1,t+1)


            with torch.no_grad():        
                pred_action = self.diffusion_sample(
                                previous_states.unsqueeze(0).float(), 
                                previous_actions.unsqueeze(0).float(),
                                episode_timesteps.unsqueeze(0).long(),
                                previous_states_padding_mask.unsqueeze(0),
                                previous_actions_padding_mask.unsqueeze(0),
                                actions_padding_mask = None,
                                max_action_len=2,
                                        )
            pred_action = pred_action.squeeze()
            played_actions = pred_action*self.actions_std
            played_actions = played_actions + self.actions_mean
            new_states_list = []
            print(len(played_actions))
            for played_action in played_actions:
                next_state, reward, terminated, truncated, info = env.step(np.array(played_action))
                new_states_list.append(next_state)
                rewards[t] = reward
                t += 1
                if terminated or truncated:
                    break

            if actions_list is None:
                actions_list = pred_action.view(-1, self.action_dimension)
            else:
                actions_list = torch.vstack((actions_list, pred_action.view(-1, self.action_dimension)))

            new_states_list = (np.array(new_states_list) - self.states_mean) / self.states_std
            states_list = torch.vstack((states_list, torch.tensor(new_states_list)))

            states_list = states_list[-num_previous_states:]
            actions_list = actions_list[-num_previous_actions:]




        


        # END STUDENT SOLUTION

        return rewards

    def sample_trajectory_gif(
    self, 
    env, 
    num_actions_to_eval_in_a_row=3, 
    num_previous_states=5,
    num_previous_actions=4, 
    ):
        """
        run a trajectory using the trained model

        Args:
            env (gym.Env): the environment to run the trajectory in
            num_actions_to_eval_in_a_row (int): the number of actions to evaluate in a row
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on

        NOTE: use with torch.no_grad(): to speed up inference by not storing gradients
        NOTE: for the first few steps, make sure to add padding to previous states/actions - use False if a state/action should be included, and True if it should be padded
        NOTE: both states and actions should be normalized before being passed to the model, and the model outputs normalized actions that need to be denormalized
        NOTE: refer to the forward function of diffusion_policy_transformer to see how to pass in the inputs (tensor shapes, etc.)
        """
        # BEGIN STUDENT SOLUTION
        frames= []
        terminated = False
        truncated = False
        state, _ = env.reset()
        frame = env.render()
        frames.append(Image.fromarray(frame))
        states_list = torch.tensor(state)
        states_list = states_list.unsqueeze(0)
        states_list = (states_list - self.states_mean) / self.states_std
        rewards = np.zeros(self.max_trajectory_length)
        t = 0
        actions_list = None
        while not terminated and not truncated:
            num_states = states_list.shape[0]
            if num_previous_states > num_states:
                previous_states = torch.vstack((states_list, torch.zeros((num_previous_states - num_states, self.state_dimension)))).float()
                if num_states == 1:
                    previous_actions = torch.zeros((num_previous_actions, self.action_dimension)).float()
                    previous_actions_padding_mask = torch.ones(num_previous_actions)
                else:
                    previous_actions = torch.vstack((actions_list, torch.zeros((num_previous_actions - num_states+1, self.action_dimension))))
                    previous_actions_padding_mask = torch.cat((torch.zeros(num_states-1), torch.ones(num_previous_states - num_states)))
                previous_states_padding_mask = torch.cat((torch.zeros(num_states), torch.ones(num_previous_states - num_states))).float()
                episode_timesteps = torch.arange(0,num_previous_states)
            else:
                previous_states = states_list
                previous_actions = actions_list
                previous_states_padding_mask = torch.zeros(num_states)
                previous_actions_padding_mask = torch.zeros(num_states-1)
                episode_timesteps = torch.arange(t-num_previous_states+1,t+1)


            with torch.no_grad():        
                pred_action = self.diffusion_sample(
                                previous_states.unsqueeze(0).float(), 
                                previous_actions.unsqueeze(0).float(),
                                episode_timesteps.unsqueeze(0).long(),
                                previous_states_padding_mask.unsqueeze(0),
                                previous_actions_padding_mask.unsqueeze(0),
                                actions_padding_mask = None,
                                max_action_len=3,
                                        )
            pred_action = pred_action.squeeze()
            played_actions = pred_action*self.actions_std
            played_actions = played_actions + self.actions_mean
            new_states_list = []
            for played_action in played_actions:
                next_state, reward, terminated, truncated, info = env.step(np.array(played_action))
                frame = env.render()
                frames.append(Image.fromarray(frame))
                new_states_list.append(next_state)
                rewards[t] = reward
                t += 1
                if terminated or truncated:
                    break

            if actions_list is None:
                actions_list = pred_action.view(-1, self.action_dimension)
            else:
                actions_list = torch.vstack((actions_list, pred_action.view(-1, self.action_dimension)))

            new_states_list = (np.array(new_states_list) - self.states_mean) / self.states_std
            states_list = torch.vstack((states_list, torch.tensor(new_states_list)))

            states_list = states_list[-num_previous_states:]
            actions_list = actions_list[-num_previous_actions:]
                



            


            # END STUDENT SOLUTION
            
        return sum(rewards), frames
    def evaluation_gif(self,
        env,
        diffusion_policy_iter=None,
        num_actions_to_eval_in_a_row=3
    ):
        """
        evaluate the model on the environment

        Args:
            diffusion_policy_iter (Optional[int]): the iteration to load the diffusion policy from
            num_samples (int): the number of samples to evaluate

        NOTE: feel free to change this function when making graphs
        """
        # load model weights:
        if diffusion_policy_iter is None:
            self.model.load_state_dict(torch.load(f"data/diffusion_policy_transformer_models/diffusion_policy.pt", map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(f"data/diffusion_policy_transformer_models/diffusion_policy_iter_{diffusion_policy_iter}.pt", map_location=self.device))


        self.model.eval()
        while(True):
            rewards, frames = self.sample_trajectory_gif(env)
            if rewards>240:
                return frames


    def evaluation(
        self,
        diffusion_policy_iter=None, 
        num_samples=20,
        num_actions_to_eval_in_a_row=3,
    ):
        """
        evaluate the model on the environment

        Args:
            diffusion_policy_iter (Optional[int]): the iteration to load the diffusion policy from
            num_samples (int): the number of samples to evaluate

        NOTE: feel free to change this function when making graphs
        """
        # load model weights:
        if diffusion_policy_iter is None:
            self.model.load_state_dict(torch.load(f"data/diffusion_policy_transformer_models/diffusion_policy.pt", map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(f"data/diffusion_policy_transformer_models/diffusion_policy_iter_{diffusion_policy_iter}.pt", map_location=self.device))

 
        self.model.eval() # turn on eval mode (this turns off dropout, running_mean, etc. that are used in training)
        traj_time = []
        rewards = np.zeros((num_samples, self.max_trajectory_length))
        os.makedirs("data/diffusion_policy_trajectories", exist_ok=True)
        for sample_trajectory in tqdm(range(num_samples)):
            time1 = time.time()
            reward = self.sample_trajectory(self.env, num_actions_to_eval_in_a_row=num_actions_to_eval_in_a_row)
            time2 = time.time()
            print(f"trajectory {sample_trajectory} took {time2 - time1} seconds")
            traj_time.append(time2 - time1)
            rewards[sample_trajectory] = reward
            print(reward)
            print(f'length of trajectory={sum(reward != 0 )}')
            print(f"rewards from trajectory {sample_trajectory}={reward.sum()}")
        print(f"average reward per trajectory={rewards.sum() / (rewards.shape[0])}")
        print(f"median reward per trajectory={np.median(rewards.sum(axis=1))}")
        print(f"max reward per trajectory={np.max(rewards.sum(axis=1))}")
        print(f"average trajectory length={np.mean(np.array([sum(1 for r in rewards[i] if r != 0) for i in range(len(rewards))]))}")
        print(f"average time to generate a trajectory={sum(traj_time) / (len(traj_time))}")

    def train(
        self, 
        num_training_steps, 
        batch_size=64, 
        print_every=5000, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        training loop that calls training_step

        Args:
            num_training_steps (int): the number of training steps to run
            batch_size (int): the batch size to use
            print_every (int): how often to print the loss
            save_every (int): how often to save the model
            wandb_logging (bool): whether to log to wandb
        """
        model = self.model
        if wandb_logging:
            wandb.init(
                name="diffusion transfomer training",
                group="diffuson transformer",
                project='walker deepRL HW3',
            )

        losses = np.zeros(num_training_steps)
        model.train()
        for training_iter in tqdm(range(num_training_steps)):
            loss = self.training_step(batch_size)
            losses[training_iter] = loss
            if wandb_logging:
                wandb.log({"loss": loss})
            if training_iter % print_every == 0:
                print(f"Training Iteration {training_iter}: loss = {loss}")
            if (training_iter + 1) % save_every == 0:
                # save model in data/diffusion_policy_transformer_models
                os.makedirs("data/diffusion_policy_transformer_models", exist_ok=True)
                torch.save(model.state_dict(), f"data/diffusion_policy_transformer_models/diffusion_policy_iter_{training_iter + 1}.pt")

        os.makedirs("data/diffusion_policy_transformer_models", exist_ok=True)
        torch.save(model.state_dict(), f"data/diffusion_policy_transformer_models/diffusion_policy.pt")
        if wandb_logging:
            wandb.finish()
        else:
            x_axis = np.arange(num_training_steps)
            plt.plot(x_axis, losses)
            plt.xlabel("Training Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss Diffusion Policy")
            plt.savefig("data/diffusion_policy_transformer_models/diffusion_policy_loss.png")
            print(f"final loss={losses[-1]}")

        return losses
    # TODO: make people write this code
    def training_step(self, batch_size):
        """
        Runs a single training step on the model.

        Args:
            batch_size (int): The batch size to use.

        NOTE: actions_padding is a mask that is False for actions to be predicted and True otherwise 
                (for instance, the model predicts 3 actions, but our batch element may contain the 2 final actions in a sequence)
                when calculating the loss, we should only consider the loss for the actions that are not padded
        NOTE: return a loss value that is a plain float (not a tensor), and is on cpu
        """
        # BEGIN STUDENT SOLUTION
        previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch = self.get_training_batch(batch_size)

        epsilon = torch.randn(actions_batch.size())
        t = torch.randint(0, self.num_train_diffusion_timesteps, size = (batch_size,1))
        noisy_actions_batch = self.training_scheduler.add_noise(actions_batch, epsilon, t)
        predictions = self.model(previous_states_batch,
                                previous_actions_batch,
                                noisy_actions_batch, 
                                episode_timesteps_batch,
                                t,
                                previous_states_padding_batch,
                                previous_actions_padding_batch,
                                actions_padding_batch,
                                )
        actions_padding_batch = actions_padding_batch.unsqueeze(-1).repeat_interleave(self.action_dimension, dim=-1)
        predictions = predictions*(1-actions_padding_batch.float())
        epsilon = epsilon*(1-actions_padding_batch.float())
        loss = nn.MSELoss()(epsilon, predictions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # END STUDENT SOLUTION

        return loss.item()

    def get_training_batch(self, batch_size, max_action_len=3, num_previous_states=5, num_previous_actions=4):
        """
        get a training batch for the model
        Args:
            batch_size (int): the batch size to use
            max_action_len (int): the maximum number of actions to predict
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on
        """
        assert num_previous_states == num_previous_actions + 1, f"num_previous_states={num_previous_states} must be equal to num_previous_actions + 1={num_previous_actions + 1}"

        # get trajectory lengths, so we can sample each trajectory with probability proportional to its length
        # this is equivalent to sampling uniformly from the set of all environment steps
        batch_indices = np.random.choice(
            np.arange(len(self.trajectory_lengths)),
            size=batch_size,
            replace=True,
            p=np.array(self.trajectory_lengths) / sum(self.trajectory_lengths)
        )

        previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch = [], [], [], [], [], [], []
        for i in range(len(batch_indices)):
            # get the start and end index for states to condition on
            end_index_state = np.random.randint(1, self.trajectory_lengths[batch_indices[i]])
            start_index_state = max(0, end_index_state - num_previous_states)

            # get the start and end index for actions to condition on (we predict the action for the final state)
            start_index_previous_actions = start_index_state
            end_index_previous_actions = end_index_state - 1

            # get the start and end index for actions to predict
            start_index_action = end_index_state
            end_index_action = min(self.trajectory_lengths[batch_indices[i]], start_index_action + max_action_len)

            previous_states = self.states[batch_indices[i], start_index_state:end_index_state]
            previous_actions = self.actions[batch_indices[i], start_index_previous_actions:end_index_previous_actions]
            actions = self.actions[batch_indices[i], start_index_action:end_index_action]

            state_dim = previous_states.shape[1]
            action_dim = actions.shape[1]

            state_seq_length = previous_states.shape[0]
            previous_action_seq_length = previous_actions.shape[0]

            # if we have less than the max number of previous states, add some padding (i.e. we're predicting a very early state)
            if state_seq_length < num_previous_states:
                previous_states = np.concatenate([previous_states, np.zeros((num_previous_states - state_seq_length, state_dim))], axis=0)
                previous_actions = np.concatenate([previous_actions, np.zeros((num_previous_actions - previous_action_seq_length, action_dim))], axis=0)
                previous_states_padding_mask = np.concatenate([np.zeros(state_seq_length), np.ones(num_previous_states - state_seq_length)], axis=0)
                previous_actions_padding_mask = np.concatenate([np.zeros(previous_action_seq_length), np.ones(num_previous_actions - previous_action_seq_length)], axis=0)
            else:
                previous_states_padding_mask = np.zeros(num_previous_states)
                previous_actions_padding_mask = np.zeros(num_previous_actions)

            # if we have less than the max number of actions, add some padding (i.e. we're predicting a very early action)
            action_seq_length = actions.shape[0]
            if action_seq_length < max_action_len:
                action_dim = actions.shape[1]
                actions = np.concatenate([actions, np.zeros((max_action_len - action_seq_length, action_dim))], axis=0)
                action_padding_mask = np.concatenate([np.zeros(action_seq_length), np.ones(max_action_len - action_seq_length)], axis=0)

            else:
                action_padding_mask = np.zeros(max_action_len)

            previous_states_batch.append(previous_states)
            previous_actions_batch.append(previous_actions)
            actions_batch.append(actions)
            episode_timesteps_batch.append(np.arange(start_index_state, start_index_state + num_previous_states)) # add extra dummy timesteps in some cases
            previous_states_padding_batch.append(previous_states_padding_mask)
            previous_actions_padding_batch.append(previous_actions_padding_mask)
            actions_padding_batch.append(action_padding_mask)

        previous_states_batch = np.stack(previous_states_batch)
        previous_actions_batch = np.stack(previous_actions_batch)
        actions_batch = np.stack(actions_batch)
        episode_timesteps_batch = np.stack(episode_timesteps_batch)
        previous_states_padding_batch = np.stack(previous_states_padding_batch)
        previous_actions_padding_batch = np.stack(previous_actions_padding_batch)
        actions_padding_batch = np.stack(actions_padding_batch)

        previous_states_batch = torch.from_numpy(previous_states_batch).float().to(self.device)
        previous_actions_batch = torch.from_numpy(previous_actions_batch).float().to(self.device)
        actions_batch = torch.from_numpy(actions_batch).float().to(self.device)
        previous_states_padding_batch = torch.from_numpy(previous_states_padding_batch).bool().to(self.device)
        previous_actions_padding_batch = torch.from_numpy(previous_actions_padding_batch).bool().to(self.device)
        actions_padding_batch = torch.from_numpy(actions_padding_batch).bool().to(self.device)
        episode_timesteps_batch = torch.from_numpy(episode_timesteps_batch).long().to(self.device)

        return previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch

def run_training():
    """
    Creates the environment, model, and optimizer, loads the data, and trains/evaluates the model using the TrainDiffusionPolicy class.
    """

    env = gym.make('BipedalWalker-v3', render_mode="rgb_array") # , render_mode="rgb_array"
    # BEGIN STUDENT SOLUTION
    ## Training

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # model = PolicyDiffusionTransformer(num_transformer_layers=6, hidden_size= 128, n_transformer_heads= 1, act_dim= action_dim, state_dim  = state_dim)
    # optim = torch.optim.AdamW(model.parameters(), lr = 0.00005, weight_decay = 0.001)
    # states_path = "/Users/vansh/Downloads/expert_collected_trajectories 2/states_BC.pkl"
    # actions_path = "/Users/vansh/Downloads/expert_collected_trajectories 2/actions_BC.pkl"
    # with open(states_path, 'rb') as f:
    #     states_array = pickle.load(f)
    # with open(actions_path, 'rb') as f:
    #     actions_array = pickle.load(f)
    # TDP = TrainDiffusionPolicy(env = env, model = model, 
    #                         optimizer = optim, 
    #                         states_array = states_array, 
    #                         actions_array = actions_array, 
    #                         device="cpu", 
    #                         num_train_diffusion_timesteps=30,
    #                         max_trajectory_length=1600
    #                     )
    # TDP.train(num_training_steps = 50000, batch_size= 256)
    # torch.save(model, 'diff_model.pth')

    ## Trajectory Code and gif
    model = torch.load('diff_model.pth')
    optim = torch.optim.AdamW(model.parameters(), lr = 0.00005, weight_decay = 0.001)
    states_path = "/Users/vansh/Downloads/expert_collected_trajectories 2/states_BC.pkl"
    actions_path = "/Users/vansh/Downloads/expert_collected_trajectories 2/actions_BC.pkl"
    with open(states_path, 'rb') as f:
        states_array = pickle.load(f)
    with open(actions_path, 'rb') as f:
        actions_array = pickle.load(f)
    TDP = TrainDiffusionPolicy(env = env, model = model, 
                            optimizer = optim, 
                            states_array = states_array, 
                            actions_array = actions_array, 
                            device="cpu", 
                            num_train_diffusion_timesteps=30,
                            max_trajectory_length=1600
                        )
    ### Trajectory
    TDP.evaluation(num_actions_to_eval_in_a_row = 3)

    ### GIF

    # model.eval()
    # env = gym.make('BipedalWalker-v3', render_mode="rgb_array")
    # frames = TDP.evaluation_gif(env)
    # frames[0].save('Diffusion_success.gif', save_all = True, append_images = frames[1:], duration = 50, loop = 0)



    

    # END STUDENT SOLUTION
if __name__ == "__main__":
    run_training()
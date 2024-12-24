import torch
from torch import nn, Tensor
import math

class PolicyDiffusionTransformer(nn.Module):

    def __init__(self, 
            num_transformer_layers, 
            state_dim,
            act_dim,
            hidden_size,
            max_episode_length=1600,
            n_transformer_heads=8,
            device="cpu",
            target="diffusion_policy",
        ):
        super(PolicyDiffusionTransformer, self).__init__()
        assert target in ["diffusion_policy", "value_model"], f"target must be either 'diffusion_policy' or 'value_model', but got {target}"
        # saving constants
        self.num_transformer_layers = num_transformer_layers
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_episode_length = max_episode_length
        self.n_transformer_heads = n_transformer_heads
        self.device = device

        # learnable epsiode timestep embedding (duration of each episode, not to be confused with diffusion noise level):
        self.episode_timestep_embedding = nn.Embedding(self.max_episode_length, self.hidden_size)

        # fixed sinusoidal timestep embeddings for diffusion noise level
        self.sinusoidal_timestep_embeddings = self.get_all_sinusoidal_timestep_embeddings(self.hidden_size, max_period=10000)
        self.sinusoidal_linear_layer = nn.Linear(self.hidden_size, self.hidden_size)

        # embed state, action, overall_return_left into hidden_size
        self.state_embedding = nn.Linear(self.state_dim, self.hidden_size)
        self.previous_act_embedding = nn.Linear(self.act_dim, self.hidden_size)
        self.act_embedding = nn.Linear(self.act_dim, self.hidden_size)

        # transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_transformer_heads,
            dim_feedforward=4*self.hidden_size,
            dropout=0.01, # 0.05
            activation='gelu',
            norm_first=True, # apply layernorm before attention (see https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm)
            batch_first=True, # batch size comes first in input tensors
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.num_transformer_layers,
        )

        # decode results into final actions
        if target == "diffusion_policy":
            self.predict_noise = nn.Sequential(
                nn.Linear(self.hidden_size, self.act_dim),
            )
        # decode results into single value for value model
        else:
            self.predict_noise = nn.Sequential(
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid(),
            )

        # set device
        self.to(self.device)

    # from https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L103
    def get_all_sinusoidal_timestep_embeddings(self, dim, max_period=10000, num_timesteps=1000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        timesteps = torch.arange(0, num_timesteps, device=self.device)
        half = dim // 2
        logs = -math.log(max_period)
        arange = torch.arange(start=0, end=half, dtype=torch.float32)
        logfreqs = logs * arange / half
        freqs = torch.exp(logfreqs).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def set_device(self, device):
        self.device = device
        self.to(device)
    
    def forward(self, 
        previous_states,
        previous_actions,
        noisy_actions,
        episode_timesteps,
        noise_timesteps,
        previous_states_mask=None,
        previous_actions_mask=None,
        actions_padding_mask=None,
        ):
        """
        forward pass of the model

        Args:
            previous_states (torch.Tensor): previous clean states, shape (batch_size, prev_states_seq_length, state_dim)
            previous_actions (torch.Tensor): previous clean actions, shape (batch_size, prev_actions_seq_length, act_dim)
            noisy_actions (torch.Tensor): noisy actions to be denoised via noise prediction, shape (batch_size, input_seq_length, act_dim)
            episode_timesteps (torch.Tensor): episode timesteps, shape (batch_size,prev_states_seq_length)
            noise_timesteps (torch.Tensor): noise timesteps for diffusion (higher timesteps implies more noisy action), shape (batch_size,)
            previous_states_mask (torch.Tensor): mask for previous states, shape (batch_size, prev_states_seq_length)
            previous_actions_mask (torch.Tensor): mask for previous actions, shape (batch_size, prev_actions_seq_length)
            actions_padding_mask (torch.Tensor): mask for noisy actions, shape (batch_size, input_seq_length)
        """

        # get batch size and action sequence length
        batch_size, input_seq_length = noisy_actions.shape[0], noisy_actions.shape[1]

        # get previous state sequence length
        prev_actions_seq_length = previous_actions.shape[1]
        prev_states_seq_length = previous_states.shape[1]

        # embed states, actions
        previous_states_embeddings = self.state_embedding(previous_states)

        previous_actions_embeddings = self.previous_act_embedding(previous_actions)

        noisy_actions_embeddings = self.act_embedding(noisy_actions)

        # get learnable episode timestep embeddings for trajectory (NOT for diffusion noise level)
        episode_timestep_embeddings = self.episode_timestep_embedding(episode_timesteps)

        # get fixed timestep embeddings for diffusion noise level
        noise_timestep_embeddings = self.sinusoidal_timestep_embeddings[noise_timesteps]

        # apply linear layer to sinusoidal timestep embeddings
        noise_timestep_embeddings = self.sinusoidal_linear_layer(noise_timestep_embeddings)

        # add episode timestep embeddings to states/previous actions ONLY (current actions are noisy, lets not mess with them)
        previous_states_embeddings = previous_states_embeddings + episode_timestep_embeddings

        previous_actions_embeddings = previous_actions_embeddings + episode_timestep_embeddings[:, 0:prev_actions_seq_length, :]

        # concatenate previous states and previous actions
        previous_observations = torch.cat((previous_states_embeddings, previous_actions_embeddings), dim=1)

        # concatenate noise timestep embeddings to previous observations
        previous_observations = torch.cat((noise_timestep_embeddings, previous_observations), dim=1)

        # get previous observations sequence length
        obs_seq_length = previous_observations.shape[1]

        # If no padding mask is given, set it to all ones.
        if previous_states_mask is None:
            previous_states_mask = torch.zeros(batch_size, prev_states_seq_length, device=self.device).bool()
        if previous_actions_mask is None:
            previous_actions_mask = torch.zeros(batch_size, prev_actions_seq_length, device=self.device).bool()
        # append a 0 to the beginning of all observations_padding_mask batches to account for the noisy timestep embeddings. 
        observations_padding_mask = torch.cat((torch.zeros(batch_size, 1, device=self.device).bool(), previous_states_mask, previous_actions_mask), dim=1)

        # if actions mask is None, set it to the ones mask with size (batch_size, input_seq_length)
        if actions_padding_mask is None:
            actions_padding_mask = torch.zeros(batch_size, input_seq_length, device=self.device).bool()

        # we add a causal mask to the input so that future noisy actions only depend on past actions in accordance with the diffuion policy paper
        # all previous observations occur before all passed in actions in the model, so no need to mask cross attention (memory_mask)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(input_seq_length, device=self.device)

        output = self.decoder(
            tgt=noisy_actions_embeddings,
            memory=previous_observations,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=actions_padding_mask,
            memory_key_padding_mask=observations_padding_mask,
        )

        # predict noise level
        noise_preds = self.predict_noise(output)
        return noise_preds

from rl.storage import Transition, RolloutStorage
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPO:
    def __init__(
            self,
            actor: torch.nn.Module,
            critic: torch.nn.Module,
            num_learning_epochs=1,
            num_mini_batches=1,
            learning_rate=1e-3,
            discount_factor=0.998,
            gae_lambda=0.95,
            value_loss_coef=1.0,
            entropy_coef=0.0,
            max_grad_norm=1.0,
            desired_kl=0.01,
            eps_clip=0.2,
            use_clipped_value_loss=True,
            schedule="fixed",
            symmetry_loss_coef=0.0,
            mirror_observation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            mirror_action_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            device='cpu',
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.device = device
        self.symmetry_loss_coef = float(symmetry_loss_coef)
        self.mirror_observation_fn = mirror_observation_fn
        self.mirror_action_fn = mirror_action_fn
        self.last_symmetry_loss = 0.0
        if self.symmetry_loss_coef < 0.0:
            raise ValueError("symmetry_loss_coef must be non-negative")
        if self.symmetry_loss_coef > 0.0 and (
                self.mirror_observation_fn is None or self.mirror_action_fn is None):
            raise ValueError("Symmetry loss requires observation and action mirror functions")


        # PPO components
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=learning_rate)
        self.transition = Transition()
        self.storage = None  # initialized later

        # PPO parameters
        self.eps_clip = eps_clip
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, critic_obs_shape, actor_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, critic_obs_shape, actor_obs_shape,
                                      action_shape, self.device)

    def act(self, obs, cri_obs):
        # Compute the actions and values
        obs = obs.to(torch.float32)
        res = self.actor(obs)
        actions, dist = res['act'].detach(), res['dist']
        self.transition.observations = obs
        self.transition.critic_obs = cri_obs
        self.transition.actions = actions
        self.transition.actions_log_prob = dist.log_prob(actions).sum(
            dim=-1).detach()  # 计算action在定义的正态分布（mean,1）中对应的概率的对数
        self.transition.action_mean = dist.mean.detach()
        self.transition.action_sigma = dist.stddev.detach()
        self.transition.values = self.critic(cri_obs).detach()
        return actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on timeouts
        if 'timeouts' in infos:
            # self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['timeouts'].unsqueeze(1).to(self.device), dim=1)
            self.transition.rewards += self.gamma * infos['timeouts'] * self.transition.values.squeeze()

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, cri_obs):
        cri_obs = cri_obs.to(torch.float32)
        with torch.no_grad():
            last_values = self.critic(cri_obs)
        self.storage.compute_returns(last_values, self.gamma, self.gae_lambda)

    def update(self):
        mean_surrogate_loss, mean_value_loss, mean_kl, mean_symmetry_loss = 0., 0., 0., 0.
        num_updates = self.num_learning_epochs * self.num_mini_batches
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, cri_obs_batch, actions_batch, target_values_batch, \
                advantages_batch, returns_batch, old_logp_batch, \
                old_mu_batch, old_sigma_batch in generator:

            res = self.actor(obs_batch)
            act, dist = res['act'], res['dist']
            # dist = self.actor(obs_batch)['dist']
            logp_batch = dist.log_prob(actions_batch).sum(dim=-1)
            mu_batch = dist.mean
            sigma_batch = dist.stddev
            entropy_batch = dist.entropy().sum(dim=-1)
            value_batch = self.critic(cri_obs_batch)
            # advantages_batch = advantages_batch / (advantages_batch.std() + 1e-5)

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(2e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = (logp_batch - old_logp_batch.squeeze()).exp()
            surr1 = ratio * advantages_batch.squeeze()
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_batch.squeeze()
            surrogate_loss = -torch.min(surr1, surr2).mean()
            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.eps_clip,
                                                                                                self.eps_clip)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            symmetry_loss = torch.zeros((), device=self.device)
            if self.symmetry_loss_coef > 0.0:
                mirrored_obs_batch = self.mirror_observation_fn(obs_batch)
                mirrored_mu_batch = self.actor(mirrored_obs_batch)['dist'].mean
                expected_mirrored_mu = self.mirror_action_fn(mu_batch)
                symmetry_loss = F.mse_loss(mirrored_mu_batch, expected_mirrored_mu)

            # total loss
            loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + self.symmetry_loss_coef * symmetry_loss
            )
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_symmetry_loss += symmetry_loss.item()
            if self.desired_kl != None and self.schedule == 'adaptive':
                mean_kl += kl_mean.item()

        self.storage.clear()
        self.last_symmetry_loss = mean_symmetry_loss / num_updates
        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates, mean_kl / num_updates

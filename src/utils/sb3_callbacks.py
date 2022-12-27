import os
import pickle
from typing import Optional

import optuna
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class TqdmCallback(BaseCallback):
    """ Callback for utilizing tqdm
    ref: https://github.com/hill-a/stable-baselines/issues/297#issuecomment-877789711
    """
    def __init__(self):
        super().__init__()
        self.progress_bar = None
    
    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class OpenLoopLickingTrackerCallback(BaseCallback):
    """ Callback for tracking a licking behavior """
    def __init__(self, env, save_path):
        super().__init__()
        self.env = env
        self.save_path = save_path

    def _on_step(self):
        return True

    def _on_training_end(self):
        with open(os.path.join(self.save_path, "lick_timing.pkl"), "wb") as f:
            pickle.dump(self.env.get_attr("lick_timing")[0], f)
        with open(os.path.join(self.save_path, "reward_history.pkl"), "wb") as f:
            pickle.dump(self.env.get_attr("reward_set")[0], f)
        with open(os.path.join(self.save_path, "skip_history.pkl"), "wb") as f:
            pickle.dump(self.env.get_attr("skip_history")[0], f)


class OpenLoopEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        n_eval_episodes=1,
        eval_freq=1,
        save_path=None,
        deterministic=True
    ):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.log_path = os.path.join(save_path, "eval")
        self.deterministic = deterministic

        self.water_spout = self.eval_env.water_spout

        self.lick_timing = []
        self.water_timing = []
        self.td_error = []
        self.q_no_lick = []
        self.q_lick = []

        self.rollout_cnt = 0

        assert self.log_path is not None, "Eval log path is not given."
        os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        self.rollout_cnt += 1
        with torch.no_grad():
            if (self.rollout_cnt % self.eval_freq) == 0:
                lick_timing_eps = []
                td_error_eps = []
                skip_steps = []
                q_no_lick_eps = []
                q_lick_eps = []

                step = 0
                done = False
                obs = self.eval_env.reset(stochasticity=False)
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).cuda()
                    next_obs, reward, done, _ = self.eval_env.step(action)
                    next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).cuda()

                    q_value_predict = self.model.q_net(obs_tensor)[0].detach().cpu()
                    q_value_target = self.model.q_net_target(next_obs_tensor)[0].detach().cpu().max()
                    q_no_lick_eps.append(q_value_predict[0].item())
                    q_lick_eps.append(q_value_predict[1].item())

                    td_error_eps.append((reward + self.model.gamma * q_value_target - q_value_predict[action]).item())
                    if action == 1:
                        lick_timing_eps.append(step)

                    obs = next_obs
                    step += 1
                    skip_steps.append(self.eval_env.skip_step)

                action_timing = np.cumsum([0] + skip_steps)
                water_x_idx = np.where(action_timing >= self.water_spout)[0][0]
                water_x = water_x_idx
                if action_timing[water_x_idx] > self.water_spout:
                    lx = self.water_spout - action_timing[water_x_idx - 1]
                    rx = action_timing[water_x_idx] - self.water_spout
                    water_x = ((water_x_idx - 1) * lx + water_x_idx * rx) / (lx + rx)

                self.lick_timing.append(lick_timing_eps)
                self.td_error.append(td_error_eps)
                self.water_timing.append(water_x)
                self.q_no_lick.append(q_no_lick_eps)
                self.q_lick.append(q_lick_eps)

        return True

    def _on_training_end(self):
        self.model.save(os.path.join(self.save_path, "best_model"))
        with open(os.path.join(self.log_path, "lick_timing.pkl"), "wb") as f:
            pickle.dump(self.lick_timing, f)
        with open(os.path.join(self.log_path, "td_error.pkl"), "wb") as f:
            pickle.dump(self.td_error, f)
        with open(os.path.join(self.log_path, "water_timing.pkl"), "wb") as f:
            pickle.dump(self.water_timing, f)
        with open(os.path.join(self.log_path, "q_no_lick.pkl"), "wb") as f:
            pickle.dump(self.q_no_lick, f)
        with open(os.path.join(self.log_path, "q_lick.pkl"), "wb") as f:
            pickle.dump(self.q_lick, f)

class InterleavedLickingTrackerCallback(BaseCallback):
    """ Callback for tracking a licking behavior """
    def __init__(self, env, n_env, save_path):
        super().__init__()
        self.env = env
        self.n_env = n_env
        self.save_path = save_path

    def _on_step(self):
        return True

    def _on_training_end(self):
        with open(os.path.join(self.save_path, f"env_history.pkl"), "wb") as f:
            pickle.dump(self.env.get_attr("env_history")[0], f)
        with open(os.path.join(self.save_path, f"env_prog_time.pkl"), "wb") as f:
            pickle.dump(self.env.get_attr("env_prog_time")[0], f)
        for i in range(self.n_env):
            with open(os.path.join(self.save_path, f"lick_timing_{i}.pkl"), "wb") as f:
                pickle.dump(self.env.get_attr("env_set")[0][i].lick_timing, f)
            with open(os.path.join(self.save_path, f"skip_history_{i}.pkl"), "wb") as f:
                pickle.dump(self.env.get_attr("env_set")[0][i].skip_history, f)


class ClosedLoopLickingTrackerCallback(BaseCallback):
    """ Callback for tracking a licking behavior """
    def __init__(self, env, save_path):
        super().__init__()
        self.env = env
        self.save_path = save_path

    def _on_step(self):
        return True

    def _on_training_end(self):
        with open(os.path.join(self.save_path, "move_timing.pkl"), "wb") as f:
            pickle.dump(self.env.get_attr("move_timing")[0], f)
        with open(os.path.join(self.save_path, "lick_pos.pkl"), "wb") as f:
            pickle.dump(self.env.get_attr("lick_pos")[0], f)

import os
import pickle
from typing import Optional

import optuna
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


class LickingTrackerCallback(BaseCallback):
    """ Callback for tracking a licking behavior """
    def __init__(self, env, save_path):
        super().__init__()
        self.env = env
        self.save_path = save_path

    def _on_step(self):
        return True

    def _on_training_end(self):
        with open(os.path.join(self.save_path, "lick_timing.pkl"), "wb") as f1:
            pickle.dump(self.env.get_attr("lick_timing")[0], f1)

        with open(os.path.join(self.save_path, "spout_timing.pkl"), "wb") as f2:
            pickle.dump(self.env.get_attr("spout_timing")[0], f2)

        with open(os.path.join(self.save_path, "actions.pkl"), "wb") as f3:
            pickle.dump(self.env.get_attr("actions")[0], f3)
        
        with open(os.path.join(self.save_path, "lick_pos.pkl"), "wb") as f4:
            pickle.dump(self.env.get_attr("lick_pos")[0], f4)
        
        with open(os.path.join(self.save_path, "move_and_lick_pos.pkl"), "wb") as f5:
            pickle.dump(self.env.get_attr("move_and_lick_pos")[0], f5)
        
        with open(os.path.join(self.save_path, "trial_start_timing.pkl"), "wb") as f6:
            pickle.dump(self.env.get_attr("trial_start_timing")[0], f6)

        with open(os.path.join(self.save_path, "shockzone_start_timing.pkl"), "wb") as f7:
            pickle.dump(self.env.get_attr("shockzone_start_timing")[0], f7)

        with open(os.path.join(self.save_path, "shockzone_end_timing.pkl"), "wb") as f8:
            pickle.dump(self.env.get_attr("shockzone_end_timing")[0], f8)

        with open(os.path.join(self.save_path, "shock_timing.pkl"), "wb") as f9:
            pickle.dump(self.env.get_attr("shock_timing")[0], f9) 

        with open(os.path.join(self.save_path, "us_type.pkl"), "wb") as f10:
            pickle.dump(self.env.get_attr("us_type")[0], f10)   
        
        with open(os.path.join(self.save_path, "time_step.pkl"), "wb") as f11:
            pickle.dump(self.env.get_attr("time_step")[0], f11)

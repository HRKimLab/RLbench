from typing import Union, List


class EpsilonScheduler(object):
    def __init__(self):
        self.step_count = 0
        self.epsilon = None

    def step(self) -> float:
        self.step_count += 1
        raise NotImplementedError

    def get_eps(self) -> float:
        return self.epsilon


class LinearDecayES(EpsilonScheduler):
    def __init__(
        self,
        init_eps: float,
        milestones: Union[int, List[int]],
        target_eps: Union[float, List[float]],
    ):
        assert (isinstance(milestones, int) and isinstance(target_eps, float)) or \
            (isinstance(milestones, list) and isinstance(target_eps, list) and len(milestones) == len(target_eps))

        super().__init__()
        self.epsilon = init_eps
        self.milestones = milestones if isinstance(milestones, list) else [milestones] 
        self.target_eps = target_eps if isinstance(target_eps, list) else [target_eps]

        self.next_milestone = self.milestones.pop(0)
        self.next_target_eps = self.target_eps.pop(0)
        self.step_eps_size = (self.epsilon - self.next_target_eps) / self.next_milestone

    def step(self) -> float:
        self.step_count += 1
        if self.step_count <= self.next_milestone:
            self.epsilon -= self.step_eps_size

        if (self.step_count == self.next_milestone) and (len(self.milestones) > 0):
            self.step_eps_size = (self.epsilon - self.target_eps[0]) / (self.milestones[0] - self.next_milestone)
            self.next_milestone = self.milestones.pop(0)
            self.next_target_eps = self.target_eps.pop(0)

        return self.epsilon


if __name__ == "__main__":
    def simulate_eps_scheduler(eps_scheduler, total_step):
        import matplotlib.pyplot as plt
        plt.plot([eps_scheduler.step() for _ in range(total_step)])
        plt.show()
    
    # Single linear decay
    eps_scheduler = LinearDecayES(
        init_eps=1.0,
        milestones=200,
        target_eps=0.2
    )
    simulate_eps_scheduler(eps_scheduler, 1000)

    # Triple linear decay
    eps_scheduler = LinearDecayES(
        init_eps=1.0,
        milestones=[25000, 100000],
        target_eps=[0.1, 0.01]
    )
    simulate_eps_scheduler(eps_scheduler, 10000000)

from tqdm import tqdm

import torch
import mlflow

from src.utils.policies import Policy


class BehaviourCloning:
    def __init__(self, policy: Policy):
        self.policy = policy

    def optimize(
        self,
        expert_trajs,
        n_steps: int = 5_000,
        batch_size: int = 64,
        lr: float = 1e-3,
        max_grad_norm: float = 1.0,
        log_every: int = 100,
    ):
        self.policy.train()

        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-6)

        for step in tqdm(range(n_steps), desc="Optimize L_outer", leave=False):
            idx = torch.randint(low=0, high=len(expert_trajs), size=(batch_size,))

            trajectory_log_probs = []
            for index in idx.tolist():
                trajectory = expert_trajs[index]
                states = trajectory["states"]
                actions = trajectory["actions"]

                if len(states) == 0:
                    continue

                log_probs = self.policy.log_prob(states, actions)  # (T,)
                trajectory_log_probs.append(log_probs.sum())

            if len(trajectory_log_probs) == 0:
                raise ValueError("The sampled batch contains only empty trajectories")

            loss = -torch.stack(trajectory_log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            else:
                grad_norm = torch.tensor(0.0)

            optimizer.step()
            scheduler.step()

            if step % log_every == 0:
                mlflow.log_metrics(
                    {
                        "behaviour_cloning/L_outer": loss.item(),
                        "behaviour_cloning/lr": optimizer.param_groups[0]["lr"],
                        "behaviour_cloning/grad_norm": float(grad_norm),
                    },
                    step=step,
                )

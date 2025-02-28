import torch
import torch.nn as nn



class RFLOW:
    def __init__(
        self,
        num_timesteps=1000,
        device='cuda'
    ):
        self.num_timesteps = num_timesteps
        self.criterion = nn.MSELoss(reduction='mean')
        self.device = device

    def sample(
        self,
        model,
        z,
        cond,
        t_seq
    ):
        eular_steps = [999,749,499,249]
        # eular_steps = [999,899,799,699,599,499,399,299,199,99]
        for step in eular_steps:
            t = torch.ones(z.shape[0],device=self.device)*step
            v = model(z, t, cond, t_seq)
            z = z + v / len(eular_steps)
        return z

    def training_losses(self, model, x_start, cond, t_seq):
        noise = torch.randn_like(x_start, device=self.device)
        B = x_start.shape[0]
        t = torch.randint(1, self.num_timesteps, (B,), device=self.device).long()
        t_norm = t.float()/(self.num_timesteps-1)
        t_norm = t_norm.view(B,1,1,1,1)
        x_t = t_norm * noise + (1 - t_norm) * x_start
        model_output = model(x_t, t, cond, t_seq)
        loss = self.criterion(x_start-noise, model_output)
        loss += self.criterion(x_start, x_t + model_output * t_norm)
        return loss
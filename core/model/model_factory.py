import os
import torch
# import logging
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from core.utils import sample
from accelerate import Accelerator
from accelerate.utils import set_seed
from core.datasets import get_datasets
from core.model.stdit import STDiTBackbone
from core.model.rectified_flow import RFLOW
# from core.helpers.metrics import Evaluator
from core.model.Autoencoder import get_model
from core.helpers.evaluation import Evaluation
from accelerate.utils import DistributedDataParallelKwargs
from core.helpers.visualization import generate_image, visualization_color


class Model(object):
    def __init__(self, configs):
        set_seed(configs.seed)
        self.configs = configs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],
                                       mixed_precision=configs.mixed_precision)
        self.network = STDiTBackbone()
        trainable_params = list(filter(lambda p: p.requires_grad, self.network.parameters()))
        optimizer = AdamW(trainable_params, 
                          lr=configs.lr,
                          betas=(configs.lr_beta1, configs.lr_beta2), 
                          weight_decay=configs.l2_norm,
                          )
        self.network, self.optimizer = self.accelerator.prepare(self.network, optimizer)
        self.scheduler = RFLOW(device=self.accelerator.device)
        self.num_timesteps = self.configs.output_length // self.configs.input_length
        
    def load(self, checkpoint_path):
        network_stats = torch.load(checkpoint_path,
                                   weights_only=True,
                                   map_location=self.accelerator.device)
        network = self.accelerator.unwrap_model(self.network)
        network.load_state_dict(network_stats)
        self.network = self.accelerator.prepare(network)
        self.accelerator.print('Model loaded from %s' % checkpoint_path)

    def train(self, frames_z):
        self.network.train()
        self.optimizer.zero_grad()
        with self.accelerator.autocast():
            loss = 0
            cond = frames_z[:,:self.configs.input_length].permute(0, 2, 1, 3, 4) # B, C, T, H, W (B, 8, 5, 16, 16)
            for i in range(1, self.num_timesteps+1):
                input_z = frames_z[:,self.configs.input_length*i:self.configs.input_length*(i+1)].permute(0, 2, 1, 3, 4)
                t_seq = torch.ones([input_z.shape[0]], device=self.accelerator.device) * i / self.num_timesteps
                loss += self.scheduler.training_losses(self.network, input_z, cond, t_seq)
                cond = input_z
        self.accelerator.backward(loss)
        self.accelerator.wait_for_everyone()
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        return loss.mean().detach().cpu().numpy()

    @torch.amp.autocast("cuda")
    def test(self, feature_dataset_loader, epoch):
        # if model.accelerator.is_main_process:
        #     logging.basicConfig(
        #             level=logging.INFO,
        #             format="%(message)s",
        #             handlers=[
        #                 logging.FileHandler('{}.log'.format(args.model_name)),
        #             ])
        # eval = Evaluator(seq_len=args.output_length,
        #                  value_scale=90.0,
        #                  thresholds=args.thresholds)
        res_path = self.configs.model_name+'_'+self.configs.datasets+'_'+epoch
        image_path = os.path.join(res_path, 'images')
        image_id = 1
        if not os.path.exists(res_path) and self.accelerator.is_main_process:
            os.makedirs(res_path)
        sample_path = os.path.join(res_path, 'samples')
        if not os.path.exists(sample_path) and self.accelerator.is_main_process:
            os.makedirs(sample_path)
        evaluater = Evaluation(seq_len=self.configs.output_length,
                               value_scale=self.configs.value_scale,
                               thresholds=self.configs.thresholds)
        gt_dataset_loader = get_datasets(name=self.configs.datasets.split("_")[0], opt='test',
                                         batch_size=self.configs.batch_size, 
                                         num_workers=self.configs.num_workers,
                                         shuffle=False)
        gt_dataset_loader = self.accelerator.prepare(gt_dataset_loader)
        test_pbar = tqdm(zip(feature_dataset_loader, gt_dataset_loader), 
                         total=len(feature_dataset_loader),
                         disable=not self.accelerator.is_main_process)
        
        autoencoder = get_model('../vae_weights_{}.pth'.format(self.configs.datasets.split("_")[0]))
        autoencoder = autoencoder.to(self.accelerator.device)
        self.network.eval()
        for itr, (frames_z, gt) in enumerate(test_pbar):
            cond = frames_z[:,:self.configs.input_length].permute(0, 2, 1, 3, 4)
            total_frames_z = []
            with torch.no_grad():
                for i in range(1, self.num_timesteps+1):
                    noise = torch.randn_like(cond, device=self.accelerator.device, dtype=torch.float32)
                    t_seq = torch.ones([noise.shape[0]], device=noise.device) * i / self.num_timesteps
                    predictions_z = self.scheduler.sample(self.network, z=noise, 
                                                          cond=cond, t_seq=t_seq)
                    cond = predictions_z
                    total_frames_z.append(predictions_z)
                self.accelerator.wait_for_everyone()
                samples = sample(torch.cat(total_frames_z, dim=2)).permute(0, 2, 1, 3, 4)
                samples = self.accelerator.gather(samples)
                B, T, C, H, W = samples.shape
                img_gen = autoencoder.decode(samples.reshape(B*T, C, H, W)) # img_gen: B*T, 3, 128, 128
                img_gen = (img_gen + 1) * 0.5
                img_gen = img_gen.reshape(B, T, -1, self.configs.img_height, self.configs.img_width).cpu().numpy()
                img_gen = np.clip(img_gen[:,:,2], 0.0, 1.0) # B, T, 128, 128
                if self.configs.datasets.split("_")[0] == "cikm":
                    img_gen = img_gen[:,:,13:-14,13:-14]
                    gt = gt[:,-self.configs.output_length:,13:-14,13:-14]
                else:
                    gt = gt[:,-self.configs.output_length:]
                gt = self.accelerator.gather(gt).cpu().numpy()
                if self.configs.visualization and self.accelerator.is_main_process:
                    visualization_color(gt[0], img_gen[0], sample_path, itr, self.configs.datasets.split("_")[0])
                if self.configs.generate_image and self.accelerator.is_main_process:
                    image_id = generate_image(img_gen, image_path, image_id, self.configs.datasets.split("_")[0])
                evaluater.update(gt.swapaxes(1, 0), img_gen.swapaxes(1, 0))
                # eval.evaluate(gt[:,:,np.newaxis], img_gen[:,:,np.newaxis])
        if self.accelerator.is_main_process:
            evaluater.save(res_path)
            # eval.done()
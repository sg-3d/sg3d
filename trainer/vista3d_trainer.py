import torch
from tqdm import tqdm

from trainer.build import BaseTrainer
from trainer.build import TRAINER_REGISTRY


@TRAINER_REGISTRY.register()
class Vista3DTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def train_step(self, epoch):
        self.model.train()
        loader = self.data_loaders["train"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                data_dict['cur_step'] = epoch * len(loader) + i
                data_dict['total_steps'] = self.total_steps
                # forward
                data_dict = self.forward(data_dict)
                # calculate loss
                loss, losses = self.loss(data_dict)
                # calculate evaluator
                metrics = self.evaluator.batch_metrics(data_dict)
                # optimize
                self.backward(loss)
                # record
                step = epoch * len(loader) + i
                losses.update(metrics)
                self.log(losses, mode="train")
                pbar.update(1)

    @torch.no_grad()
    def eval_step(self, epoch):
        self.model.eval()
        loader = self.data_loaders["val"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            data_dict = {k : v for k, v in data_dict.items() if isinstance(v, torch.Tensor)}
            data_dict = self.accelerator.gather_for_metrics(data_dict)
            self.evaluator.update(data_dict)
            pbar.update(1)
        is_best, results = self.evaluator.record()
        self.log(results, mode="val")
        self.evaluator.reset()
        return is_best

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        loader = self.data_loaders["test"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            self.evaluator.update(data_dict)
            pbar.update(1)
        is_best, results = self.evaluator.record()
        self.log(results, mode="test")
        self.evaluator.reset()
        return results

    def run(self):
        if self.mode == "train":
            start_epoch = self.exp_tracker.epoch
            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                self.train_step(epoch)

                # if self.accelerator.is_main_process:
                is_best = self.eval_step(epoch)
                self.accelerator.print(f"[Epoch {epoch + 1}] finished eval, is_best: {is_best}")

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    if is_best:
                        self.save("best.pth")
                    if (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_{epoch+1}.pth")
        else:
            return self.test_step()
        self.accelerator.end_training()

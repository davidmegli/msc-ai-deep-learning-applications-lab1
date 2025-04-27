import os
import time
import copy
import torch
import wandb
import shutil
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable, Dict, Optional

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        output_dir: str = './output',
        max_epochs: int = 100,
        patience: int = 10,
        mixed_precision: bool = False,
        project_name: str = 'training',
        use_wandb: bool = True,
        run_name: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.patience = patience
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.mixed_precision = mixed_precision

        os.makedirs(self.output_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard"))

        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=project_name, name=run_name or f"run_{int(time.time())}")
            wandb.watch(self.model, log="all")

        self.best_loss = np.inf
        self.epochs_no_improve = 0
        self.start_epoch = 0
        self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss

    def save_checkpoint(self, epoch, best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss
        }
        filename = os.path.join(self.output_dir, 'best_checkpoint.pth' if best else 'last_checkpoint.pth')
        torch.save(state, filename)
        if best:
            print(f"Best checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, best=True):
        filename = os.path.join(self.output_dir, 'best_checkpoint.pth' if best else 'last_checkpoint.pth')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No checkpoint found at {filename}")

        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from {filename}")

    def train(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"Epoch {epoch+1}/{self.max_epochs}")

            train_loss = self.train_one_epoch()

            val_loss = None
            if self.val_loader:
                val_loss = self.validate()

            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar('Loss/val', val_loss, epoch)

            if self.use_wandb:
                metrics = {'train_loss': train_loss}
                if val_loss is not None:
                    metrics['val_loss'] = val_loss
                wandb.log(metrics, step=epoch)

            if self.scheduler:
                self.scheduler.step()

            # Checkpointing
            self.save_checkpoint(epoch, best=False)
            if val_loss is not None and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, best=True)
                self.epochs_no_improve = 0
            elif val_loss is not None:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.writer.close()
        if self.use_wandb:
            wandb.finish()
'''
Author: David Megli
Date: 2025-04-28
'''
# trainer.py

import os
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler,
                 device, output_dir, max_epochs, patience, mixed_precision, project_name,
                 use_wandb, run_name):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.project_name = project_name
        self.use_wandb = use_wandb
        self.run_name = run_name

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

        # early stopping
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.start_epoch = 0

        if self.use_wandb:
            wandb.init(project=self.project_name, name=self.run_name, config={})
            wandb.watch(self.model)

        # Auto-Resume
        self.checkpoint_path = os.path.join(self.output_dir, 'last_checkpoint.pth')
        if os.path.exists(self.checkpoint_path):
            print(f"[INFO] Found checkpoint. Resuming training from {self.checkpoint_path}")
            self.load_checkpoint()

    def save_checkpoint(self, epoch):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']

    def train(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"Epoch {epoch+1}/{self.max_epochs}")

            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            # Save learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            if self.use_wandb:
                wandb.log({"LearningRate": current_lr})

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log losses
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            if self.use_wandb:
                wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})

            # Early Stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            else:
                self.early_stop_counter += 1

            self.save_checkpoint(epoch)

            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

        return running_loss / len(self.val_loader.dataset)
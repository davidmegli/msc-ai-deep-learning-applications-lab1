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
import time
from torch.utils.data import DataLoader
from typing import Callable, Optional

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
        resume: bool = False,
    ):
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
        self.resume = resume

        self.scaler = torch.amp.GradScaler('cuda',enabled=self.mixed_precision)
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

        # early stopping
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.start_epoch = 0

        self.total_params = self.count_parameters(self.model)
        print(f"[INFO] Model has {self.total_params} trainable parameters.")
        self.writer.add_text('Model/ParameterCount', str(self.total_params))

        if self.use_wandb:
            wandb.init(project=self.project_name, name=self.run_name, config={})
            wandb.watch(self.model)
            wandb.config.update({"Total_parameters": self.total_params})

        # Auto-Resume
        self.checkpoint_path = os.path.join(self.output_dir, 'last_checkpoint.pth')
        if os.path.exists(self.checkpoint_path) and self.resume:
            print(f"[INFO] Found checkpoint. Resuming training from {self.checkpoint_path}")
            self.load_checkpoint()

    def save_checkpoint(self, epoch):
        """Save model checkpoint with epoch and timestamp."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'best_loss': self.best_loss
        }
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        filename = f'checkpoint_epoch_{epoch}_{timestamp}.pth'
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
        torch.save(checkpoint, os.path.join(self.output_dir, 'last_checkpoint.pth'))

        self.cleanup_old_checkpoints(max_keep=5)


    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']

    def cleanup_old_checkpoints(self, max_keep=0):
        """Delete old checkpoints, keeping only the last `max_keep`."""
        if max_keep == 0:
            return

        checkpoint_files = sorted(
            [f for f in os.listdir(self.output_dir)
            if f.startswith('checkpoint_epoch') and f.endswith('.pth')],
            key=lambda x: int(x.split('_')[2])  # Assuming filename like checkpoint_epoch_{epoch}_{timestamp}.pth
        )

        if len(checkpoint_files) > max_keep:
            files_to_remove = checkpoint_files[:-max_keep]
            for f in files_to_remove:
                try:
                    os.remove(os.path.join(self.output_dir, f))
                    print(f"[INFO] Removed old checkpoint: {f}")
                except Exception as e:
                    print(f"[WARNING] Could not remove {f}: {e}")

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train(self):
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"Epoch {epoch+1}/{self.max_epochs}")

            train_loss, train_accuracy, val_loss, val_accuracy = self.train_one_epoch(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
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

            if self.patience > 0 and self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        self.writer.close()
        return train_losses, train_accuracies, val_losses, val_accuracies

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.max_epochs}", unit="batch")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = correct / total

        # VALIDATE
        val_loss, val_accuracy = self.validate(epoch)

        # SCHEDULER STEP
        if self.scheduler:
            self.scheduler.step()

        # LEARNING RATE
        current_lr = self.optimizer.param_groups[0]['lr']

        # LOGGING
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        self.writer.add_scalar('LearningRate', current_lr, epoch)

        if self.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
                "epoch": epoch
            })

        print(f"Epoch [{epoch}/{self.max_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}")

        return train_loss, train_accuracy, val_loss, val_accuracy


    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(self.val_loader)
        val_accuracy = correct / total

        return val_loss, val_accuracy
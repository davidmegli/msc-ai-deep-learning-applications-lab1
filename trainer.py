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
        if os.path.exists(self.checkpoint_path):
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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
        self.writer.close()

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(self.train_loader)

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
        self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        self.writer.add_scalar('LearningRate', current_lr, epoch)

        if self.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
                "epoch": epoch
            })

        print(f"Epoch [{epoch}/{self.num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}")

        return train_loss, val_loss, val_accuracy


    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
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
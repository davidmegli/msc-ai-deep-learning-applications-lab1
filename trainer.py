'''
Author: David Megli
Date: 2025-04-28
File: trainer.py
Description: Trainer class for model training and evaluation
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
import re

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
        config: Optional[dict] = None):

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
        self.config = config or {}
        basename = "" + self.model.__class__.__name__ + "_" + self.optimizer.__class__.__name__
        self.basename = self.config['trainer'].get('run_name', basename) + "_"

        self.progressive_unfreeze = False
        self.unfreeze_schedule = []

        self.maybe_freeze_backbone()

        self.scaler = torch.amp.GradScaler('cuda',enabled=self.mixed_precision)
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

        # early stopping
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.start_epoch = 0

        self.total_params = self.count_parameters(self.model)
        print(f"[INFO] Model has {self.total_params} trainable parameters.")
        self.writer.add_text('Model/ParameterCount', str(self.total_params))

        # Resume
        if self.resume:
            # If a checkpoint is given, try to load it
            pretrained_path = self.config['trainer'].get('pretrained_checkpoint')
            self.checkpoint_path = pretrained_path
            if pretrained_path and os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location=self.device)

                # Gestione caricamento modello con classi di output diverse
                model_dict = self.model.state_dict()
                checkpoint_dict = checkpoint['model_state_dict']

                # Verifica se è un caso speciale: dimensione del fc layer diversa
                fc_weight_key = 'fc.weight'
                fc_bias_key = 'fc.bias'

                fc_mismatch = (
                    fc_weight_key in checkpoint_dict
                    and fc_bias_key in checkpoint_dict
                    and checkpoint_dict[fc_weight_key].shape[0] != config['model']['params']['num_classes']
                )
                # If there is a mismatch in fully connected dimensions
                if fc_mismatch:
                    print("[INFO] FC layer mismatch: loading all layers except the classifier (fc)")

                    # Escludi fc.weight e fc.bias
                    pretrained_dict = {
                        k: v for k, v in checkpoint_dict.items()
                        if k in model_dict and not k.startswith('fc.')
                    }
                    # Load the rest of the network, without fc. The fc should be randomly initialized and finetuned
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
                # If there is no size mismatch between the fc layers
                else:
                    # Normal loading of the full network
                    self.model.load_state_dict(checkpoint_dict)

                #self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[INFO] Loaded pretrained weights from {pretrained_path}")
            
            else:
                filename = self.basename + 'last_checkpoint.pth'
                self.checkpoint_path = os.path.join(self.output_dir, filename)
                if os.path.exists(self.checkpoint_path):
                    print(f"[INFO] Found checkpoint. Resuming training from {self.checkpoint_path}")
                    self.load_checkpoint()

        if self.use_wandb:
            wandb.init(project=self.project_name, name=self.run_name, config={})
            wandb.watch(self.model)
            wandb.config.update({"Total_parameters": self.total_params})


    def maybe_freeze_backbone(self):
        # Setup modalità di freeze
        freeze_mode = self.config.get('trainer', {}).get('freeze_backbone', False)
        if freeze_mode == "progressive":
            self.progressive_unfreeze = True
            self.unfreeze_schedule = self.config['trainer'].get('unfreeze_at_epochs', [])

            # Congela inizialmente tutto tranne FC
            for name, param in self.model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False
            print("[INFO] Model backbone frozen (progressive mode)")
        elif freeze_mode is True:
            for name, param in self.model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False
            print("[INFO] Model backbone frozen.")
        else:
            print("[INFO] Training full model (no freezing).")

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
        filename = self.basename + f'checkpoint_epoch_{epoch}_{timestamp}.pth'
        last_checkpoint_name = self.basename + 'last_checkpoint.pth'
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
        torch.save(checkpoint, os.path.join(self.output_dir, last_checkpoint_name))

        self.cleanup_old_checkpoints(max_keep=5)

    def save_as_best_model(self, epoch):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'best_loss': self.best_loss
        }
        filename = self.basename + 'best_model.pth'
        torch.save(checkpoint, os.path.join(self.output_dir, filename))

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def log_gradient_norms(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient Norm: {total_norm}")
        if self.use_wandb:
            wandb.log({'gradient_norm': total_norm})

    def unfreeze_layers_progressively(self, epoch):
        if not hasattr(self.model, 'blocks'):
            return

        unfreeze_epochs = self.unfreeze_schedule
        for e in unfreeze_epochs:
            if epoch == e:
                idx = unfreeze_epochs.index(e)
                # Sblocca i blocchi dalla fine (i più alti)
                blocks_to_unfreeze = 1
                block_indices = list(range(len(self.model.blocks) - blocks_to_unfreeze - idx, len(self.model.blocks)))
                print(f"[INFO] Unfreezing blocks at epoch {epoch}: {block_indices}")
                for i in block_indices:
                    for param in self.model.blocks[i].parameters():
                        param.requires_grad = True
    
    def cleanup_old_checkpoints(self, max_keep=0):
        """Delete old checkpoints, keeping only the last `max_keep`."""
        if max_keep == 0:
            return

        checkpoint_files = [
            f for f in os.listdir(self.output_dir)
            if f.startswith(self.basename + 'checkpoint_epoch_') and f.endswith('.pth')
        ]

        # Estrai il numero di epoca da ciascun filename usando regex
        def extract_epoch(filename):
            match = re.search(r'checkpoint_epoch_(\d+)_', filename)
            return int(match.group(1)) if match else -1

        # Ordina i file in base all'epoca
        checkpoint_files.sort(key=extract_epoch)

        if len(checkpoint_files) > max_keep:
            files_to_remove = checkpoint_files[:-max_keep]
            for f in files_to_remove:
                try:
                    os.remove(os.path.join(self.output_dir, f))
                    print(f"[INFO] Removed old checkpoint: {f}")
                except Exception as e:
                    print(f"[WARNING] Could not remove {f}: {e}")
                    
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
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early Stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.early_stop_counter = 0
                self.save_as_best_model(epoch)
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
        if self.progressive_unfreeze:
            self.unfreeze_layers_progressively(epoch)

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
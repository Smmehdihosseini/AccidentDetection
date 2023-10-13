import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import os
import time
from thop import profile 
from datetime import datetime


class Trainer:
    def __init__(self, model,
                 train_loader,
                 val_loader,
                 device="cuda:0",
                 num_epochs=200,
                 batch_verbose=10,
                 lr=0.001,
                 weight_decay=0.01,
                 max_lr=0.001,
                 pct_start=0.1,
                 early_stopping_limit=10,
                 anneal_strategy='cos',
                 base_checkpoint_dir="model_checkpoints"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.batch_verbose = batch_verbose
        self.num_epochs = num_epochs
        self.early_stopping_limit = early_stopping_limit
        self.base_checkpoint_dir = base_checkpoint_dir
        self.anneal_strategy = anneal_strategy

        self.model.to(self.device)

        # Prepare optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

        # Implementing learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=pct_start,  # 10% of training steps for warm-up
            anneal_strategy=self.anneal_strategy,
        )

        self.criterion = nn.CrossEntropyLoss()

        # Calculate FLOPs and number of parameters
        self.calculate_model_complexity()

    def calculate_model_complexity(self):
        # Ensure model is in evaluation mode for static graph
        self.model.eval()
        # Get the input dimension from the train_loader
        input_tensor = next(iter(self.train_loader))[0]
        input_tensor = input_tensor.to(self.device)

        # Calculate FLOPs
        macs, params = profile(self.model, inputs=(input_tensor,))
        # Convert to FLOPs (multiply by 2, as one MAC = 2 FLOPs)
        flops = macs * 2

        print(f"- Model FLOPs: {flops}")
        print(f"- Model Parameters: {params}")

        # Return model back to training mode
        self.model.train()

    def train_epoch(self):
        
        self.model.train()
        all_losses = []

        for i, (images, labels_accident) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels_accident = labels_accident.to(self.device)

            # Forward pass
            labels_pred = self.model(images)
            loss_accident = self.criterion(labels_pred, labels_accident)

            # Backward pass
            self.optimizer.zero_grad()
            loss_accident.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Record loss
            all_losses.append(loss_accident.item())

            # Print loss every 10 batches
            if i % self.batch_verbose == 0:
                print(f"- Batch {i}/{len(self.train_loader)}, Current batch loss: {loss_accident.item()}")

            # Step the scheduler
            self.scheduler.step()

        avg_loss = np.mean(all_losses)
        return avg_loss

    def validate_epoch(self):

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for images, labels_accident in self.val_loader:
                images = images.to(self.device)
                labels_accident = labels_accident.to(self.device)

                # Forward pass
                severity_pred = self.model(images)
                loss_accident = self.criterion(severity_pred, labels_accident)

                val_losses.append(loss_accident.item())

        avg_val_loss = np.mean(val_losses)
        return avg_val_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self):

        best_val_loss = float('inf')
        early_stopping_counter = 0

        start_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        current_checkpoint_dir = os.path.join(self.base_checkpoint_dir, start_time_str)
        os.makedirs(current_checkpoint_dir, exist_ok=True)

        # Initialize timer
        start_time = time.time()

        for epoch in range(self.num_epochs):
            print("-----------------------------------------")
            print(f"Starting Epoch {epoch + 1}/{self.num_epochs}")
            print("-----------------------------------------")

            # Start epoch timer
            epoch_start_time = time.time()

            avg_loss = self.train_epoch()
            avg_val_loss = self.validate_epoch()
            print(f"+ Average Training/Validation Loss: {avg_loss:.4f}/{avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(self.base_checkpoint_dir, f"best_model_epoch_{epoch + 1}.pth")
                self.save_model(best_model_path)
                print(f"* Saved New Best Model At {best_model_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.early_stopping_limit:
                    print("! Early Stopping Condition Met. Stopping Training.")
                    break

            # End epoch timer and calculate epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print("........................................................")
            print(f"||| Epoch {epoch + 1}/{self.num_epochs} Finished: {epoch_time:.2f} Seconds |||")

        # End timer
        end_time = time.time()


        final_model_path = os.path.join(current_checkpoint_dir, "final_model.pth")
        self.save_model(final_model_path)
        print(f"\n+ Training Finished! Final Model Saved To {final_model_path}")

        total_training_time = end_time - start_time
        print(f"+ Total Training Time: {total_training_time:.2f} Seconds")

import os
import torch
from tqdm import tqdm
from loguru import logger
from .metrics import MetricsTracker
import json

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, config, checkpoint_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        self.best_qwk = -1.0
        self.patience_counter = 0
        self.early_stopping_patience = config.get('early_stopping', 5)
        self.num_classes = config.get('num_classes', 5)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_one_epoch(self):
        self.model.train()
        tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            if len(batch) == 3:
                images, labels, _ = batch
                ehrs = None
            else:
                images, ehrs, labels, _ = batch
                ehrs = ehrs.to(self.device)
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                if ehrs is not None:
                    outputs = self.model(image=images, ehr=ehrs)['logits']
                else:
                    outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            if self.scheduler is not None:
                self.scheduler.step()
                
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            tracker.update(labels.cpu().numpy(), preds.cpu().numpy(), probs.detach().cpu().numpy(), loss.item())
            pbar.set_postfix({'loss': loss.item()})
            
        return tracker.compute(self.num_classes)
        
    def validate(self):
        self.model.eval()
        tracker = MetricsTracker()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if len(batch) == 3:
                    images, labels, _ = batch
                    ehrs = None
                else:
                    images, ehrs, labels, _ = batch
                    ehrs = ehrs.to(self.device)
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if ehrs is not None:
                    outputs = self.model(image=images, ehr=ehrs)['logits']
                else:
                    outputs = self.model(images)
                    
                loss = self.loss_fn(outputs, labels)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                tracker.update(labels.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy(), loss.item())
                
        return tracker.compute(self.num_classes)
        
    def train(self, num_epochs):
        history = {'train': [], 'val': []}
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Check unfreeze
            if hasattr(self.model, 'freeze_epochs') and epoch == self.model.freeze_epochs:
                logger.info("Unfreezing backbone")
                self.model.unfreeze_backbone()
                
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()
            
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)
            
            logger.info(f"Train Loss: {train_metrics['loss']:.4f} | QWK: {train_metrics['qwk']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f} | QWK: {val_metrics['qwk']:.4f}")
            
            if val_metrics['qwk'] > self.best_qwk:
                self.best_qwk = val_metrics['qwk']
                self.save_checkpoint(epoch, val_metrics)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        return history
        
    def save_checkpoint(self, epoch, metrics):
        path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

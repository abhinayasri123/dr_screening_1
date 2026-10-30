import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from ..training.metrics import compute_all_metrics

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

class DRFedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        epochs = config.get("epochs", 1)
        lr = config.get("lr", 1e-4)
        mu = config.get("mu", 0.0)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        global_params = [torch.tensor(p).to(self.device) for p in parameters] if mu > 0 else None
        
        self.model.train()
        for _ in range(epochs):
            for batch in self.train_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                    ehrs = None
                else:
                    images, ehrs, labels, _ = batch
                    ehrs = ehrs.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                if ehrs is not None:
                    outputs = self.model(image=images, ehr=ehrs)['logits']
                else:
                    outputs = self.model(images)
                    
                loss = self.criterion(outputs, labels)
                
                # FedProx proximal term
                if mu > 0:
                    proximal_term = 0.0
                    for local_weights, global_weights in zip(self.model.parameters(), global_params):
                        proximal_term += (local_weights - global_weights).norm(2)
                    loss += (mu / 2) * proximal_term
                    
                loss.backward()
                optimizer.step()
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        y_true = []
        y_pred = []
        y_proba = []
        
        with torch.no_grad():
            for batch in self.val_loader:
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
                    
                batch_loss = self.criterion(outputs, labels).item()
                loss += batch_loss
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_proba.extend(probs.cpu().numpy())
                
        loss /= len(self.val_loader)
        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        
        # flwr requires scalar metrics
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(metrics["accuracy"]), "qwk": float(metrics["qwk"])}

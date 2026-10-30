import torch
import numpy as np

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

class MCDropoutEstimator:
    def __init__(self, model, n_passes=20, device='cuda'):
        self.model = model
        self.n_passes = n_passes
        self.device = device
        self.model.to(self.device)
        
    def enable_mc_dropout(self, model):
        """Enables dropout layers in eval mode."""
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    def predict_with_uncertainty(self, image_tensor, ehr_tensor=None):
        self.model.eval()
        self.enable_mc_dropout(self.model)
        
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        if ehr_tensor is not None:
            if len(ehr_tensor.shape) == 1:
                ehr_tensor = ehr_tensor.unsqueeze(0)
            ehr_tensor = ehr_tensor.to(self.device)
            
        probas = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                if ehr_tensor is not None:
                    out = self.model(image=image_tensor, ehr=ehr_tensor)
                else:
                    try:
                        out = self.model(image_tensor)
                    except TypeError:
                        out = self.model(image=image_tensor)
                logits = out['logits'] if isinstance(out, dict) else out
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probas.append(probs)
                
        probas = np.array(probas) # [n_passes, batch_size, num_classes]
        
        mean_proba = np.mean(probas, axis=0)[0]
        std_proba = np.std(probas, axis=0)[0]
        
        predicted_class = int(np.argmax(mean_proba))
        confidence = float(mean_proba[predicted_class])
        
        # Predictive Entropy
        entropy = -np.sum(mean_proba * np.log(mean_proba + 1e-12))
        
        # Expected Entropy
        expected_entropy = -np.mean(np.sum(probas[:, 0, :] * np.log(probas[:, 0, :] + 1e-12), axis=1))
        
        mutual_information = float(entropy - expected_entropy)
        
        return {
            'mean_proba': mean_proba,
            'std_proba': std_proba,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'entropy': float(entropy),
            'mutual_information': mutual_information
        }
        
    def classify_uncertainty(self, std):
        """Classifies uncertainty based on std deviation of predicted class."""
        mean_std = np.mean(std)
        if mean_std < 0.05:
            return 'Low'
        elif mean_std < 0.15:
            return 'Medium'
        return 'High'

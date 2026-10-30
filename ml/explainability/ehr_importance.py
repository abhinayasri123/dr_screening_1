import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

class EHRImportanceExplainer:
    def __init__(self, model, ehr_encoder, feature_names):
        self.model = model
        self.ehr_encoder = ehr_encoder
        self.feature_names = feature_names
        
    def explain_with_shap(self, ehr_tensor, background_data=None):
        if background_data is None:
            background_data = torch.zeros((100, ehr_tensor.shape[1])).to(ehr_tensor.device)
            
        def model_wrapper(x):
            x_t = torch.tensor(x, dtype=torch.float32).to(ehr_tensor.device)
            with torch.no_grad():
                if hasattr(self.model, 'mode') and self.model.mode == 'ehr_only':
                    return self.model(ehr=x_t)['logits'].cpu().numpy()
                elif hasattr(self.model, 'mode') and self.model.mode == 'multimodal':
                     # Mock image input for shap if multimodal
                     dummy_img = torch.zeros((x_t.shape[0], 3, 300, 300)).to(ehr_tensor.device)
                     return self.model(image=dummy_img, ehr=x_t)['logits'].cpu().numpy()
                else:
                    return self.model(x_t).cpu().numpy()

        explainer = shap.KernelExplainer(model_wrapper, background_data.cpu().numpy())
        shap_values = explainer.shap_values(ehr_tensor.cpu().numpy())
        
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            importance_dict[name] = np.mean(np.abs(shap_values[0][:, i]))
            
        return importance_dict
        
    def explain_with_gradients(self, ehr_tensor):
        ehr_tensor = ehr_tensor.clone().requires_grad_(True)
        if hasattr(self.model, 'mode') and self.model.mode == 'multimodal':
            dummy_img = torch.zeros((ehr_tensor.shape[0], 3, 300, 300)).to(ehr_tensor.device)
            outputs = self.model(image=dummy_img, ehr=ehr_tensor)['logits']
        else:
            outputs = self.model(ehr=ehr_tensor)['logits'] if hasattr(self.model, 'mode') else self.model(ehr_tensor)
            
        max_logits, _ = torch.max(outputs, dim=1)
        max_logits.sum().backward()
        
        grads = torch.abs(ehr_tensor.grad).mean(dim=0).cpu().numpy()
        importance_dict = {name: val for name, val in zip(self.feature_names, grads)}
        return importance_dict
        
    def plot_importance(self, importance_dict, save_path=None):
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_items]
        vals = [x[1] for x in sorted_items]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(names, vals)
        ax.set_title('EHR Feature Importance')
        ax.set_xlabel('Importance Score')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        return fig

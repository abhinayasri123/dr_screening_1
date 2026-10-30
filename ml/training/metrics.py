import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

def compute_quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def compute_all_metrics(y_true, y_pred, y_proba, num_classes=5, class_names=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['qwk'] = compute_quadratic_weighted_kappa(y_true, y_pred)
    
    try:
        metrics['macro_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except Exception:
        metrics['macro_auc'] = 0.0
        
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    metrics['confusion_matrix'] = cm.tolist()
    
    per_class_sens = []
    per_class_spec = []
    per_class_prec = []
    per_class_f1 = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        per_class_sens.append(float(sens))
        per_class_spec.append(float(spec))
        per_class_prec.append(float(prec))
        per_class_f1.append(float(f1))
        
    metrics['per_class_sensitivity'] = per_class_sens
    metrics['per_class_specificity'] = per_class_spec
    metrics['per_class_precision'] = per_class_prec
    metrics['per_class_f1'] = per_class_f1
    metrics['macro_precision'] = float(np.mean(per_class_prec))
    metrics['macro_recall'] = float(np.mean(per_class_sens))
    metrics['class_distribution'] = [int(np.sum(cm[i, :])) for i in range(num_classes)]
    
    return metrics

class MetricsTracker:
    def __init__(self):
        self.reset()
        
    def update(self, y_true_batch, y_pred_batch, y_proba_batch, loss):
        self.y_true.extend(y_true_batch)
        self.y_pred.extend(y_pred_batch)
        self.y_proba.extend(y_proba_batch)
        self.losses.append(loss)
        
    def compute(self, num_classes=5):
        metrics = compute_all_metrics(self.y_true, self.y_pred, self.y_proba, num_classes)
        metrics['loss'] = np.mean(self.losses)
        return metrics
        
    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_proba = []
        self.losses = []

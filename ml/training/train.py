import os
import argparse
import json
import torch
import torch.optim as optim
from .trainer import Trainer
from .losses import get_loss_function
from ..models.cnn_model import DRClassifier
from ..models.multimodal_model import MultimodalDRClassifier
from ..preprocessing.image_preprocessor import create_data_loaders

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

def load_config(config_dir="configs"):
    # Dummy config for standalone execution
    return {
        'num_classes': 5,
        'early_stopping': 5,
        'learning_rate': 1e-4,
        'backbone_lr': 1e-5,
        'weight_decay': 1e-4,
        'loss': 'cross_entropy'
    }

def setup_data(config, data_dir, batch_size):
    return create_data_loaders(data_dir, batch_size, num_workers=4)

def setup_model(config, num_classes, device, mode):
    if mode == 'cnn':
        model = DRClassifier(num_classes=num_classes)
    else:
        model = MultimodalDRClassifier(num_classes=num_classes, mode=mode)
    return model.to(device)

def setup_optimizer(model, config):
    # Differentiate LR for backbone vs head if it's DRClassifier
    if isinstance(model, DRClassifier):
        optimizer = optim.AdamW([
            {'params': model.backbone.features.parameters(), 'lr': config['backbone_lr']},
            {'params': model.backbone.classifier.parameters(), 'lr': config['learning_rate']}
        ], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    return optimizer

def setup_scheduler(optimizer, config, num_epochs, steps_per_epoch):
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[param_group['lr'] * 10 for param_group in optimizer.param_groups],
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cnn', choices=['cnn', 'multimodal', 'ehr_only'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--logs-dir', type=str, default='logs')
    args = parser.parse_args()

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader, weights = setup_data(config, args.data_dir, args.batch_size)
    if train_loader is None:
        print("Data not found, exiting.")
        return

    model = setup_model(config, config['num_classes'], device, args.mode)
    optimizer = setup_optimizer(model, config)
    
    steps_per_epoch = len(train_loader)
    scheduler = setup_scheduler(optimizer, config, args.epochs, steps_per_epoch)
    
    loss_fn = get_loss_function(config['loss'], weights, config['num_classes'])
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, config, args.checkpoint_dir)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    history = trainer.train(args.epochs)
    
    # Final evaluation on test set
    trainer.load_checkpoint(os.path.join(args.checkpoint_dir, 'best_model.pth'))
    # Swap val_loader for test_loader in validate
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()
    
    os.makedirs(args.logs_dir, exist_ok=True)
    with open(os.path.join(args.logs_dir, 'training_results.json'), 'w') as f:
        json.dump({'history': history, 'test_metrics': test_metrics}, f, indent=4)

if __name__ == '__main__':
    main()

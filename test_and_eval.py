
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from loss_functions import ABNNLoss, CustomMAPLoss

class ECE:
    def __init__(self, bins=10):
        self.bins = bins

    def measure(self, probs, labels):
        probs = np.array(probs)
        labels = np.array(labels)
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, self.bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        return ece

def test_model_with_metrics(loss_fn: nn.Module, model: nn.Module, test_loader, load_path: str = 'vit_mnist.pth',
               calculate_uncert: bool = False, calculate_nll_loss: bool = False, calculate_ece_error: bool = False,
               calculate_auprc: bool = False, calculate_auc_roc: bool = False, calculate_fpr_95: bool = False, 
               count_params: bool = False, plot_uncert: bool = False, predict_uncert: bool = False, 
               model_class: type = None, models: list = None, num_samples: int = 40, num_classes: int = 10,
               Weight_decay: float = 5e-4) -> None:
    """
    Evaluates a model on a test dataset and computes various metrics.

    Args:
        loss_fn (nn.Module): Loss function (or "ABNN" for ABNNLoss).
        model (nn.Module): The model to evaluate.
        test_loader: DataLoader for the test dataset.
        load_path (str): Path to the pre-trained model weights.
        calculate_uncert (bool): Whether to calculate uncertainty per class.
        calculate_nll_loss (bool): Whether to calculate NLL loss (not implemented).
        calculate_ece_error (bool): Whether to calculate Expected Calibration Error (ECE).
        calculate_auprc (bool): Whether to calculate AUPRC.
        calculate_auc_roc (bool): Whether to calculate AUC-ROC.
        calculate_fpr_95 (bool): Whether to calculate FPR at 95% recall.
        count_params (bool): Whether to count model parameters.
        plot_uncert (bool): Whether to plot uncertainty (not implemented).
        predict_uncert (bool): Whether to use multiple stochastic passes for predictions.
        model_class (type): Class of the model (required if predict_uncert=True).
        models (list): List of model state dictionaries (required if predict_uncert=True).
        num_samples (int): Number of stochastic forward passes for uncertainty prediction.
        num_classes (int): Number of classes in the dataset.
        Weight_decay (float): Weight decay parameter (not used in this implementation).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(load_path), strict=False)
    model.to(device)
    model.eval()

    # Set up the loss criterion
    if loss_fn == "ABNN":
        eta = torch.ones(num_classes)
        criterion = ABNNLoss(num_classes, model.parameters()).to(device)
    elif loss_fn == "CustomMAPLoss":
        eta = torch.ones(num_classes)
        criterion = CustomMAPLoss(eta, model.parameters()).to(device)
    else:
        criterion = loss_fn

    all_predictions = []
    all_labels = []
    test_loss = 0.0
    all_probs = []
    uncertainties = {i: [] for i in range(num_classes)} if calculate_uncert else None

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if predict_uncert:
                if model_class is None or models is None:
                    raise ValueError("model_class and models must be provided for uncertainty prediction.")

                # Ensemble with majority voting across models and stochastic passes
                batch_predictions = []
                for model_state_dict in models:
                    net = model_class(norm_type = 'bnl')
                    net.load_state_dict(model_state_dict, strict = True)
                    net.to(device)
                    net.eval()

                    # Run multiple stochastic forward passes
                    model_preds = []
                    for _ in range(num_samples):
                        outputs = net(inputs)
                        _, preds = torch.max(outputs, 1)
                        model_preds.append(preds)
                    # Take mode of predictions for this model
                    stacked_preds = torch.stack(model_preds)
                    mode_preds, _ = torch.mode(stacked_preds, dim=0)
                    batch_predictions.append(mode_preds)

                # Take mode across all models
                stacked_batch_preds = torch.stack(batch_predictions)
                final_preds, _ = torch.mode(stacked_batch_preds, dim=0)
            else:
                # Single forward pass without uncertainty prediction
                outputs = model(inputs)
                _, final_preds = torch.max(outputs, 1)

            # Compute loss
            outputs = model(inputs)  # Recompute outputs for loss and metrics
            test_loss += criterion(outputs, labels).item()

            # Collect predictions and labels
            all_predictions.append(final_preds.cpu())
            all_labels.append(labels.cpu())

            # Compute probabilities for other metrics
            if calculate_ece_error or calculate_auprc or calculate_auc_roc or calculate_fpr_95:
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

            # Compute uncertainty if requested
            if calculate_uncert:
                mc_outputs = torch.stack([model(inputs) for _ in range(num_samples)])
                probabilities = torch.softmax(mc_outputs, dim=-1)
                variance = probabilities.var(dim=0)
                for i in range(num_classes):
                    class_mask = (labels == i)
                    if class_mask.any():
                        class_variance = variance[class_mask, i].mean().item()
                        uncertainties[i].append(class_variance)
                        
    

    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute accuracy
    correct = (all_predictions == all_labels).sum().item()
    total = all_labels.size(0)
    accuracy = 100 * correct / total

    # Compute other metrics
    test_loss /= len(test_loader)
    all_preds_np = all_predictions.numpy()
    all_targets_np = all_labels.numpy()
    f1 = f1_score(all_targets_np, all_preds_np, average='weighted')

    avg_uncertainties = {i: np.mean(uncertainties[i]) if uncertainties[i] else 0 for i in range(num_classes)} if calculate_uncert else None

    all_probs = np.concatenate(all_probs, axis=0) if all_probs else None
    ece_score = ECE(bins=10).measure(all_probs, all_targets_np) if calculate_ece_error else None

    if calculate_auprc:
        auprs = []
        all_labels_one_hot = np.eye(num_classes)[all_targets_np]
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(all_labels_one_hot[:, i], all_probs[:, i])
            aupr = auc(recall, precision)
            auprs.append(aupr)
        mean_aupr = np.mean(auprs)
    else:
        mean_aupr = None

    if calculate_auc_roc:
        aucs = []
        all_labels_one_hot = np.eye(num_classes)[all_targets_np]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        mean_auc = np.mean(aucs)
    else:
        mean_auc = None

    if calculate_fpr_95:
        fpr_95_recall = []
        all_labels_one_hot = np.eye(num_classes)[all_targets_np]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], all_probs[:, i])
            idx = np.where(tpr >= 0.95)[0][0]
            fpr_at_95_recall = fpr[idx]
            fpr_95_recall.append(fpr_at_95_recall)
        mean_fpr_95_recall = np.mean(fpr_95_recall)
    else:
        mean_fpr_95_recall = None

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) if count_params else None

    # Print results
    print(f'\nTest set Metrics:')
    print(f'  Average loss: {test_loss:.4f}')
    print(f'  F1 Score: {f1:.4f}')
    print(f'  Accuracy: {accuracy:.2f}%')
    
    if calculate_uncert:
        print(f'  Uncertainties: {avg_uncertainties}')
    if calculate_ece_error:
        print(f'  ECE: {ece_score:.4f}')
    if calculate_auprc:
        print(f'  Mean AUPR: {mean_aupr:.4f}')
    if calculate_auc_roc:
        print(f'  Mean AUC: {mean_auc:.4f}')
    if calculate_fpr_95:
        print(f'  Mean FPR at 95% Recall: {mean_fpr_95_recall:.4f}')
    if count_params:
        print(f'  Number of Parameters: {param_count}')

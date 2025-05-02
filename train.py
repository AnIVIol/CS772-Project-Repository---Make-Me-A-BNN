import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary
from loss_functions import ABNNLoss, CustomMAPLoss
import time
import matplotlib.pyplot as plt

def train_model(model: nn.Module, trainloader: DataLoader, testloader: DataLoader, epochs:int =30,
                loss_type: str ='ABNNLoss', learning_rate: float =0.01,
                momentum: float =0.9, weight_decay: float =5e-4, BNL_enable: bool = False, save_path: str = './trained_models/default.pth',
                load_path: str = './trained_models/default.pth', debug: bool = False, strictness: bool = True, num_classes: int = 10, random_prior: bool = False) -> (list, list):
    
    if loss_type == 'ABNNLoss':
        loss_func = ABNNLoss(num_classes, model.parameters())
    elif loss_type == 'CrossEntropyLoss':
        loss_func = nn.CrossEntropyLoss()
    elif loss_type == 'CustomMAPLoss':
        eta = torch.ones(num_classes)
        if random_prior:
            eta = torch.distributions.Dirichlet(torch.ones(num_classes)).sample() * num_classes
        loss_func = CustomMAPLoss(eta, model.parameters())
    else:
        raise ValueError("Unsupported loss type. Choose either 'ABNNLoss', 'CrossEntropyLoss', or 'CustomMAPLoss'.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if BNL_enable:
        state_dict = torch.load(load_path, map_location = device)
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k}
        
        keep_running_stats_layers = ['norm1', 'ln3', 'ln4', 'norm5']

        # # Filter the state dictionary
        # filtered_state_dict = {}
        # for k, v in state_dict.items():
        #     # Split the key into layer name and parameter name
        #     layer_name, param_name = k.split('.', 1)
            
        #     # Check if the parameter is a running statistic
        #     if param_name in ['running_mean', 'running_var', 'num_batches_tracked']:
        #         # Only keep the statistic if the layer is in keep_running_stats_layers
        #         if layer_name in keep_running_stats_layers:
        #             filtered_state_dict[k] = v
        #     else:
        #         # Keep all other parameters (e.g., weight, bias) regardless of layer
        #         filtered_state_dict[k] = v
        
        model.load_state_dict(filtered_state_dict, strict=strictness)
        print("BNL model loaded from {}".format(load_path))
        print('Model weights loaded.')
        
    print(device)
    model.to(device)
    loss_func.to(device)
    summary(model, (3, 32, 32))
    
    start_time = time.time()
    
    print('Start Training')
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            eta = torch.rand(labels.size(0), device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_losses.append(loss.item())
        if debug:
            print(f'[Epoch {epoch + 1}, Loss: {running_loss}')
        running_loss = 0.0

    end_time = time.time()

    print('Finished Training')
    print(f'Time taken to train the model: {end_time - start_time:.2f} seconds')
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total:.2f} %')
    torch.save(model.state_dict(), save_path)
    print('Saved the model weights')
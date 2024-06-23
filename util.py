import torch
import numpy as np

def train(model,optimizer,train_loader,device,epoch,arg,
                     loss_function,warmup=True):
    model.train()
    lr_scheduler = None
    if epoch == 0 and warmup is True:  # Only in the first run need to be wraumup
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    epoch_loss = []
    step = 0
    for batch_data in train_loader:
        step += 1
        images, labels = batch_data
        
        if isinstance(images, list):
            images = [image.to(device) for image in images]
        else:
            images = images.to(device)        
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        # print(f"{i + 1}/{sequence} sequence,{step} step train_loss: {loss.item():.4f}")
    if lr_scheduler is not None:
        lr_scheduler.step()

    now_lr = optimizer.param_groups[0]["lr"]
    mean_loss = np.mean(epoch_loss)

    return mean_loss,now_lr



def val(model,val_loader,arg,epoch,device,loss_function):
    model.eval()
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    step = 0
    with torch.no_grad():
        for val_data in val_loader:
            step += 1
            images, labels = val_data
            
            if isinstance(images, list):
                images = [image.to(device) for image in images]
            else:
                images = images.to(device)    
            labels = labels.to(device)
            y_pred = torch.cat([y_pred, model(images)], dim=0)
            y = torch.cat([y, labels], dim=0)

        loss = loss_function(y_pred,y)
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc = acc_value.sum().item() / len(acc_value)
  
    mean_loss = loss.item() / step

    return mean_loss,acc


def test(model,val_loader,arg,epoch,device,loss_function):
    model.eval()
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    step = 0
    with torch.no_grad():
        for val_data in val_loader:
            step += 1
            images, labels = val_data
            if isinstance(images, list):
                images = [image.to(device) for image in images]
            else:
                images = images.to(device)    
            labels = labels.to(device)
            y_pred = torch.cat([y_pred, model(images)], dim=0)
            y = torch.cat([y, labels], dim=0)

        loss = loss_function(y_pred,y)
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc = acc_value.sum().item() / len(acc_value)
  
    mean_loss = loss.item() / step

    return mean_loss,acc

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    Create a learning rate scheduler with warmup for the optimizer.

    """

    def f(x):
        """Return a learning rate multiplier factor based on the step number."""
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
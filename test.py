import torch

def test(model, test_loader, cuda):
    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs, _ = model(images, alpha=0)
            _, predicted = torch.max(outputs, 1)
            n_total += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    accuracy = 100 * n_correct / n_total
    return accuracy

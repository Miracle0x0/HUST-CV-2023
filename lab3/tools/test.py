import torch
import torch.nn as nn
import numpy as np

num_classes = 10
cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def check_accuracy(model, loader):
    """检验模型

    Args:
        model (nn.Module): 待检验的模型
        loader (_type_): 测试数据集

    Returns:
        在测试集上的平均准确率 acc, 并输出模型在每一类上的准确率.
    """
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_correct = 0
    num_samples = 0
    model.eval()  # 模型置为验证状态
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            if not loader.dataset.train:
                c = (preds == y).squeeze()
                for i in range(len(y)):
                    label = y[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        if not loader.dataset.train:
            print('Class\tAccuracy')
            print('----------------')
            for i in range(num_classes):
                print(f'{cifar10_classes[i]:5s} \t |  {100 * class_correct[i] / class_total[i]}')

        return acc

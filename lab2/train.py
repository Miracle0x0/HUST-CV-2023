import sys
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from functools import wraps
from configs import num_classes


def print_progress(cur_iter, total_iter, message, prefix='', suffix='', decimals=1, bar_len=50, print_every=50):
    cur_iter += 1
    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (cur_iter / float(total_iter)))
    filled_len = int(round(bar_len * cur_iter) / float(total_iter))
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    if (cur_iter - 1) % print_every == 0:
        msg_str = "\tIter " + str(cur_iter - 1) + ", loss = " + "{:.4f}".format(message)
    else:
        msg_str = ""
    sys.stdout.write('\r%s |%s| %s%s %s%s' % (prefix, bar, percent, '%', msg_str, suffix))
    if cur_iter == total_iter:
        sys.stdout.write('\n')
    sys.stdout.flush()


def timer(func):
    """计时函数

    Args:
        func (Any): 待计时的函数

    Returns:
        Any: 计算结果
    """

    @wraps(func)
    def wrap(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        h, remainder = divmod((end_time - start_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        print(f"\n{func.__name__} time: {h}:{m}:{s}")
        return result

    return wrap


@timer
def train(model: nn.Module,
          criterion,
          optimizer,
          scheduler,
          loader_train,
          loader_val,
          epochs: int = 1,
          verbose: bool = True,
          print_every: int = 50,
          save_dir=None):
    """训练模型

    Args:
        model (nn.Module): 待训练的模型
        criterion (_type_): 损失函数
        optimizer (_type_): 优化器
        scheduler (_type_): 学习率动态调整方案
        loader_train (_type_): 训练数据集
        loader_val (_type_): 验证数据集
        epochs (int, optional): epoch 大小. 默认为 1.
        verbose (bool, optional): 是否输出详细信息. 默认为 True.
        print_every (int, optional): 控制输出频率. 默认为 50.
        save_dir (_type_, optional): 模型保存路径, 如指定则将模型保存到该位置. 默认为 None.

    Returns:
        best_model, train_loss_history, train_acc_history, val_acc_history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    dtype = torch.float32

    train_loss_history = []
    train_acc_history, val_acc_history = [], []

    best_acc = 0.0
    best_model = None

    for e in range(epochs):
        train_loss = 0.0
        num_correct = 0
        num_samples = 0
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            loss = criterion(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            print_progress(t, len(loader_train), message=loss.item(), print_every=print_every)
            # if t % print_every == 0:
            #     print('Iter %d, loss = %.4f' % (t, loss.item()), end='\r')

        train_loss /= len(loader_train)
        train_loss_history.append(train_loss)
        scheduler.step(train_loss)

        train_acc = float(num_correct) / num_samples
        train_acc_history.append(train_acc)

        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for t, (x, y) in enumerate(loader_val):
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)
                scores = model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

            val_acc = float(num_correct) / num_samples
            val_acc_history.append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model

            if verbose:
                print(f"Epoch {e + 1} / {epochs}\ttrain_acc = {train_acc:.4f}, val_acc = {val_acc:.4f}")

        if save_dir:
            torch.save(best_model, save_dir)

    return best_model, train_loss_history, train_acc_history, val_acc_history


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

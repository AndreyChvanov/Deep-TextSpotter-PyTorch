import torch
import numpy as np
import torch.nn as nn
import multiprocessing
from torch.autograd import Variable
import torch.nn.functional as F


def calc_alpha(output, label, S, T, blank):
    a = np.zeros((S, T))
    a[0, 0] = output[blank, 0]
    try:
        a[1, 0] = output[label[0], 0]
    except:
        print()
    c = np.sum(a[:, 0])
    if c > 0:
        a[:, 0] = a[:, 0] / c
    for t in range(1, T):
        start = max(0, S - 2 * (T - t))
        end = min(2 * t + 2, S)
        for s in range(start, end):
            i = max(0, (s - 1) // 2)
            if s % 2 == 0:
                if s == 0:
                    a[s, t] = a[s, t - 1] * output[blank][t]
                else:
                    a[s, t] = (a[s, t - 1] + a[s - 1, t - 1]) * output[blank, t]
            elif s == 1 or label[i] == label[i - 1]:
                a[s, t] = (a[s, t - 1] + a[s - 1, t - 1]) * output[label[i], t]
            else:
                a[s, t] = (a[s, t - 1] + a[s - 1, t - 1] + a[s - 2, t - 1]) * output[label[i], t]

        c = np.sum(a[start:end, t])
        if c > 0:
            a[start:end, t] = a[start:end, t] / c
    return a


def ctc_loss(args):
    output, label, lengths, blank = args
    softmax_grad = np.zeros_like(output)
    if output.shape[-1] == 0 or lengths == 0:
        return 0.0, softmax_grad
    s, t = len(label) * 2 + 1, lengths
    output = output[:, :t]
    a = calc_alpha(output, label, s, t, blank)
    b_ = calc_alpha(np.fliplr(output), label[::-1], s, t, blank)
    b = np.flipud(np.fliplr(b_))

    ab = a * b
    lab = np.zeros(output.shape)
    for s in range(s):
        if s % 2 == 0:
            lab[blank, :] += ab[s, :]
            ab[s, :] = ab[s, :] / output[blank, :]
        else:
            l = max(0, (s - 1) // 2)
            lab[label[l], :] += ab[s, :]
            ab[s, :] = ab[s, :] / output[label[l], :]

    lh = np.sum(ab, axis=0)
    loss = -np.sum(np.log(lh))
    softmax_grad[:, :t] = output - lab / (output * lh)
    nan_indexes = np.isnan(softmax_grad)
    softmax_grad[nan_indexes] = 0.0
    return loss, softmax_grad


class CTCLoss(nn.Module):
    def __init__(self, deice):
        super().__init__()
        self.grad = None
        self.device = deice
        self.pool = multiprocessing.Pool(4)

    def forward(self, preds, labels, lengths, blank_index=0):
        data = [(preds[i].detach().cpu().numpy(), labels[i], lengths[i], blank_index) for i in range(len(preds))]
        result = self.pool.map(ctc_loss, data)
        # result = [ctc_loss(item )for item in data]
        self.grad = [result[i][1] for i in range(len(result))]
        loss_per_element = [result[i][0] for i in range(len(result))]
        batch_loss = np.mean(loss_per_element)
        return batch_loss

    def backward(self):
        return Variable(torch.tensor(self.grad)).to(self.device), None


if __name__ == '__main__':
    from cnn import CNN

    model = CNN(in_channels=1, vocab_size=34)
    x = torch.randn((2, 1, 500, 32))
    out = model(x).squeeze(-1)
    blank = 33
    label = [[1, 3, 3, 7, ], [1, 3, 3, 7, 5]]
    lengths = [500, 500]
    lengths = tuple([np.floor(l / 4).astype(np.int32) for l in lengths])
    criterion = CTCLoss(deice='cpu')
    out = F.softmax(out, dim=-1)
    loss = criterion(out, label, lengths, blank_index=blank)
    print(loss)


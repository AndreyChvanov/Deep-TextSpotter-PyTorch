import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datasets.mjSynth_dataset import MjSynthDataset, collage_fn
from models.cnn import CNN
from models.loss import CTCLoss
from torch.autograd import Variable
from datasets.utils import decode_path, decode_sequence
import Levenshtein
from models.logger import NeptuneLogger


class Trainer:
    def __init__(self, dataset_config, train_config, logger_cfg):
        self.dataset_config, self.train_config = dataset_config, train_config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.softmax = nn.Softmax(dim=1)
        self.criterion = CTCLoss(self.device)
        self.__init_datasets()
        self.__init_dataloader()
        self.__init_model()
        self.__init_optimizer()
        self.logger = NeptuneLogger(logger_cfg)

    def __init_datasets(self):
        self.train_dataset = MjSynthDataset(self.dataset_config)
        self.test_dataset = MjSynthDataset(self.dataset_config, train=False, mean=self.train_dataset.mean)

    def __init_dataloader(self):
        self.train_loader = DataLoader(self.train_dataset, collate_fn=collage_fn,
                                       batch_size=self.train_config.batch_size, drop_last=True,
                                       shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, collate_fn=collage_fn, batch_size=self.train_config.batch_size,
                                      drop_last=False, shuffle=False)

    def __init_model(self):
        self.model = CNN(in_channels=1, vocab_size=self.train_dataset.vocab_capacity)
        self.model.to(self.device)
        self.model.train()

    def __init_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9)

    def fit(self):
        for epoch in range(self.train_config.nrof_epochs):
            # if epoch == self.train_config.nrof_epochs - 1 or epoch == 1:
            self.eval(epoch)
            self.model.train()
            for i, batch_data in enumerate(self.train_loader):
                x, gt, ori_sizes = batch_data
                self.optimizer.zero_grad()
                x = Variable(x).to(self.device)
                logits = self.model(x)
                outputs = self.softmax(logits)
                ori_sizes = tuple([np.floor(size / 4).astype(np.int32) for size in ori_sizes])
                loss = self.criterion(outputs, gt, ori_sizes)
                grad = self.criterion.backward()
                logits.backward(grad)
                self.optimizer.step()
                batch_gt_seqs = decode_sequence(gt)
                model_output = decode_path(outputs.detach().cpu().numpy())
                batch_pred_seqs = decode_sequence(model_output)
                acc = np.sum([
                    1 - (Levenshtein.distance(pred, gt) / max(len(pred), len(gt))) for pred, gt in
                    zip(batch_pred_seqs, batch_gt_seqs)
                ]) / len(batch_gt_seqs)
                self.logger.add_scalar('train/batch_loss', loss, epoch * len(self.train_loader) + i)
                self.logger.add_scalar('train/batch_accuracy', acc, epoch * len(self.train_loader) + i)
                # print('accuracy: ', acc)
                # print('predict: ', batch_pred_seqs)
                # print('gt: ', batch_gt_seqs)
                # print('_____')

    def eval(self, epoch):
        all_acc = []
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(self.test_loader):
                x, gt, ori_sizes = batch_data
                x = Variable(x).to(self.device)
                logits = self.model(x)
                outputs = self.softmax(logits)
                ori_sizes = tuple([np.floor(size / 4).astype(np.int32) for size in ori_sizes])
                loss = self.criterion(outputs, gt, ori_sizes)
                batch_gt_seqs = decode_sequence(gt)
                model_output = decode_path(outputs.cpu().numpy())
                batch_pred_seqs = decode_sequence(model_output)
                acc = np.sum([
                    1 - (Levenshtein.distance(pred, gt) / max(len(pred), len(gt))) for pred, gt in
                    zip(batch_pred_seqs, batch_gt_seqs)
                ]) / len(batch_gt_seqs)
                all_acc.append(acc)
                losses.append(loss.item())
                if i % 100 == 0:
                    print(i, acc)
        all_acc = np.array(all_acc)
        losses = np.array(losses)
        mean_acc = all_acc.mean()
        mean_loss = losses.mean()
        self.logger.add_scalar('test/loss', mean_loss, epoch)
        self.logger.add_scalar('test/accuracy', mean_acc, epoch)


if __name__ == '__main__':
    from configs.dataset_config import cfg as dataset_config
    from configs.train_config import cfg as train_config
    from configs.logger_config import cfg as logger_cfg

    trainer = Trainer(dataset_config, train_config, logger_cfg)
    trainer.fit()

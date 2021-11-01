import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from tqdm import tqdm
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if __name__ == '__main__':
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():  # args.test_only = True的话，只进行测试，并返回True
            t.train()
            t.test()

        checkpoint.done()


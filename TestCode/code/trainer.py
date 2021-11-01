import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\n-------------------Evaluations({})-------------------'.format(self.args.model))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.ckp.write_log('############[dataset={}]---[scale={}]############\n'.format(self.args.testset,scale))
                avg_psnr = 0
                avg_ssim = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    sr = self.model(lr, idx_scale)

                    # visuals = utility.get_current_visual(lr, hr, sr, self.args.rgb_range)
                    # psnr, ssim = utility.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
        
                    save_list = [utility.quantize(sr, self.args.rgb_range)]
                    if not no_eval:
                        visuals = utility.get_current_visual(lr, hr, sr, self.args.rgb_range)
                        psnr, ssim = utility.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
                        avg_psnr += psnr
                        avg_ssim += ssim
                        self.ckp.write_log("{}:\tpsnr = {} ,\tssim = {} .".format(filename.ljust(30,' '), psnr,ssim))

                    if self.args.save_results:
                        self.ckp.save_results_nopostfix(filename, save_list, scale)

                avg_psnr = avg_psnr / len(self.loader_test)
                avg_ssim = avg_ssim / len(self.loader_test)
                print(avg_psnr, avg_ssim)
                self.ckp.write_log("{}:\tpsnr = {} ,\tssim = {} .".format('AVG'.ljust(30,' '), avg_psnr, avg_ssim))
               
        self.ckp.write_log(
            'Total time: {:.2f}s, ave time: {:.2f}s\n'.format(timer_test.toc(), timer_test.toc()/len(self.loader_test)), refresh=True
        )
    
    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs


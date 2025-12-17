import torch
import torch.optim as optim
from tqdm import trange
import os
from valid import valid
from loss import MatchLoss
from utils import tocuda
from tensorboardX import SummaryWriter
# tensorboard --logdir=../model --host=127.0.0.1 --port=6006

from thop import profile
from thop import clever_format
from torchstat import stat

def train_step(step, optimizer, model, match_loss, data, config):
    model.train()
    balance = config.loss_balance 
    if step >= 80000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.999996

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # flops, params = profile(model.to(device),inputs=(data,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)

    # # print('Peak memory(MB): ',torch.cuda.max_memory_allocated()/1e6)
    res_logits, res_e_hat, load_balance_loss = model(data)
    # load balance loss
    loss = balance * load_balance_loss
    loss_val = []
    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss]
    optimizer.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        # print(name)
        if torch.any(torch.isnan(param.grad)):
            print('skip because nan')
            return loss_val
    loss_val.append(load_balance_loss)
    optimizer.step()
    return loss_val


def train(model, train_loader, valid_loader, config):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    match_loss = MatchLoss(config)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    writer=SummaryWriter(os.path.join(config.log_path, 'log_file'))
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        best_acc = -1
        start_epoch = 0
    train_loader_iter = iter(train_loader)
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        loss_vals = train_step(step, optimizer, model, match_loss, train_data, config)
        if step%config.log_intv==0:
            writer.add_scalar('lr', cur_lr, step)
            writer.add_scalar('EssentionLoss', loss_vals[0], step)
            writer.add_scalar('ClassifyLoss', loss_vals[1], step)
            writer.add_scalar('RegressionLoss', loss_vals[2], step)
            writer.add_scalar('load_balance_loss', loss_vals[3], step)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss,  _, _, _, loss_load_balance = valid(valid_loader, model, step, config)
            writer.add_scalar('val_EssentionLoss', geo_loss, step)
            writer.add_scalar('val_ClassifyLoss', cla_loss, step)
            writer.add_scalar('val_RegressionLoss', l2_loss, step)
            writer.add_scalar('loss_load_balance', l2_loss, step)
            writer.add_scalar('val_acc', va_res, step)            
            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))

        # if b_save:
        #     torch.save({
        #     'epoch': step + 1,
        #     'state_dict': model.state_dict(),
        #     'best_acc': best_acc,
        #     'optimizer' : optimizer.state_dict(),
        #     }, checkpoint_path)

        if b_save:
                    checkpoint_filename = f"checkpoint_step{step}.pth"
                    checkpoint_filepath = os.path.join(config.log_path, checkpoint_filename)
                    torch.save({
                        'epoch': step + 1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                    }, checkpoint_filepath)

       
import torch
import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter
from tqdm import tqdm
from decoder import decoder
from vgg import vgg
from SANet import Net
import torch.backends.cudnn
from Infloader import train_transform, FlatFolderDataset, InfiniteSamplerWrapper, FlatFolderDataset2
from apex import amp


def adjust_learning_rate(optimizer, iteration_count, lr, lr_decay):
    """Imitating the original implementation"""
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    batchsize = 2
    device = torch.device('cuda')
    vgg_pth = r'./vgg_normalised.pth'
    decoder = decoder()
    vgg = vgg()
    start_iter = 18000
    content_dir = r'F:/XiyuUnderGradThesis/data/train_img'
    style_dir = r'F:/XiyuUnderGradThesis/data/cropped_cz_mask'
    lr = 1e-4
    max_iter = 100000
    content_weight = 2.
    style_weight = 3.
    save_model_interval = 2000
    torch.backends.cudnn.benchmark = True

    vgg.load_state_dict(torch.load(vgg_pth))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    network = Net(vgg, decoder, start_iter)
    network.decoder.load_state_dict(torch.load(r'.\checkpoints_save\decoder_iter_62000.pth'))
    network.transform.load_state_dict(torch.load(r'.\checkpoints_save\transformer_iter_18000.pth'))
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset2(content_dir, content_tf)
    style_dataset = FlatFolderDataset(style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=batchsize,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=4))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=batchsize,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=4))

    optimizer = torch.optim.Adam([
        {'params': network.decoder.parameters()},
        {'params': network.transform.parameters()}], lr=lr)

    network, optimizer = amp.initialize(network, optimizer, opt_level="O1")

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batchsize}')
    if (start_iter > 0):
        optimizer.load_state_dict(torch.load(r'.\checkpoints_save\optimizer_iter_62000.pth'))
    global_step = 0

    with tqdm(total=max_iter - start_iter,
              desc=global_step / (max_iter - start_iter),
              unit='batch') as pbar:
        running_avg_loss = 0
        for i in range(start_iter, max_iter):
            adjust_learning_rate(optimizer, iteration_count=i, lr=lr, lr_decay=5e-5)
            content_images, content_mask = next(content_iter)#.to(device)
            content_images = content_images.to(device)
            content_mask = content_mask.to(device)
            style_images = next(style_iter).to(device)
            loss_c, loss_s, l_identity1, l_identity2, road_identity = network(content_images, style_images, content_mask)
            loss_c = 1 * loss_c
            loss_s = 2 * loss_s
            l_identity1 *= 150
            l_identity2 *= 1
            road_identity *= 500
            loss = loss_c + loss_s + l_identity1 + l_identity2 + road_identity

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

            optimizer.zero_grad()
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            running_avg_loss += loss.item()
            pbar.update(1)
            if global_step % 100 == 0:
                writer.add_scalar('loss', running_avg_loss / 100, global_step)
                running_avg_loss = 0

            # if global_step % 1000 == 0:
            #     for tag, value in network.decoder.named_parameters():
            #         tag = tag.replace('.', '/')
            #         # writer.add_histogram('dec_weights/' + tag, value.data.cpu().numpy(), global_step)
            #         # writer.add_histogram('dec_grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            #     writer.add_scalar('dec_learning_rate', optimizer.param_groups[0]['lr'], global_step)
            #     for tag, value in network.transform.named_parameters():
            #         tag = tag.replace('.', '/')
            #         # writer.add_histogram('trans_weights/' + tag, value.data.cpu().numpy(), global_step)
            #         # writer.add_histogram('trans_grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            #     writer.add_scalar('trans_learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
                state_dict = decoder.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict,
                           '{:s}/decoder_iter_{:d}.pth'.format(r'./checkpoints_save',
                                                               i + 1))
                state_dict = network.transform.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict,
                           '{:s}/transformer_iter_{:d}.pth'.format(r'./checkpoints_save',
                                                                   i + 1))
                state_dict = optimizer.state_dict()
                torch.save(state_dict,
                           '{:s}/optimizer_iter_{:d}.pth'.format(r'./checkpoints_save',
                                                                 i + 1))


import argparse
import os
from utils import get_config, _write_images
import torch
from data import create_dataset, create_dataloader
from models.networks import define_G
from data.util import tensor2img
import skimage.io as sio
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba-hq-regular_list.yaml', help="net configuration")
parser.add_argument('--output_folder', type=str, default='outputs/celebahq-regular/saved_images', help="output image path")
parser.add_argument('--checkpoint', type=str, default='outputs/celebahq-regular/checkpoints/latest_G.pth',
                    help="checkpoint of generator")
opts = parser.parse_args()

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

device = torch.device('cuda')
# Setup model and data loader


model = define_G(config).to(device)
model.load_state_dict(torch.load(opts.checkpoint), strict=True)
model.eval()

print('Loading the checkpoint for G [{:s}] ...'.format(opts.checkpoint))

with torch.no_grad():
    dataset_opt = config['datasets']['test']
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))

    # Start testing

    for index, test_data in enumerate(test_loader):
        v_input, v_output, v_target = [], [], []
        visual_images = []
        var_input, var_mask, var_target, img_paths = test_data['input'], test_data['mask'], test_data['target'], \
                                                     test_data['paths']
        var_input = var_input.to(device)
        var_mask = var_mask.to(device)
        var_target = var_target.to(device)
        var_output = var_mask.detach() * model(torch.cat([var_input, var_mask], dim=1)) + (
                1 - var_mask.detach()) * var_input.detach()
        v_input.append(var_input.detach()[0].float().cpu())
        v_output.append(var_output.detach()[0].float().cpu())
        v_target.append(var_target.detach()[0].float().cpu())
        visual_images.extend(v_input)
        visual_images.extend(v_output)
        visual_images.extend(v_target)
        _write_images(visual_images, 1, '%s/%s' % (opts.output_folder, img_paths[0].split('/')[-1]))
        saved_mask = (var_mask.detach()[0].float().cpu().numpy().squeeze() * 255).round().astype(np.uint8)
        saved_input = (var_mask.detach()[0].float().cpu() + ((v_target[0] + 1) / 2)).numpy().squeeze().transpose(1, 2, 0).clip(0, 1)
        saved_output = tensor2img(v_output)
        saved_target = tensor2img(v_target)
        sio.imsave(os.path.join(opts.output_folder, 'mask', img_paths[0].split('/')[-1].split('.')[0] + '.png'), saved_mask)
        sio.imsave(os.path.join(opts.output_folder, 'input', img_paths[0].split('/')[-1]), saved_input)
        sio.imsave(os.path.join(opts.output_folder, 'output', img_paths[0].split('/')[-1]), saved_output[0])
        sio.imsave(os.path.join(opts.output_folder, 'target', img_paths[0].split('/')[-1]), saved_target[0])


print('End of testing.')

import argparse
import os
from tensorboardX import SummaryWriter
from utils import get_config, prepare_sub_folder, _write_images, write_html
from data import create_dataset, create_dataloader
import math
from models.inpainting_model import InpaintingModel
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/paris-celeba-hq-regular_list.yaml', help='path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help='output path of the tensorboard file')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = get_config(args.config)
torch.backends.cudnn.benchmark = True

# tensorboard
model_name = os.path.splitext(os.path.basename(args.config))[0].split('_')[0]
train_writer = SummaryWriter(os.path.join(args.output_path + '/logs', model_name))

output_dir = os.path.join(args.output_path + '/outputs', model_name)
checkpoint_dir, image_dir = prepare_sub_folder(output_dir)
config['checkpoint_dir'] = checkpoint_dir

# create train and val dataloader
for phase, dataset_opt in config['datasets'].items():
    if phase == 'train':
        train_set = create_dataset(dataset_opt)
        train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
        print('Number of training images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        total_iters = int(dataset_opt['n_iter'])
        total_epochs = int(math.ceil(total_iters / train_size))
        print('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
        train_loader = create_dataloader(train_set, dataset_opt)
    elif phase == 'val':
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt)
        print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
    elif phase == 'test':
        pass
    else:
        raise NotImplementedError('Unsupported phase: {:s}'.format(phase))

# create model
model = InpaintingModel(config)

start_epoch = 0
current_step = 0

# training
for epoch in range(start_epoch, total_epochs):
    for _, train_data in enumerate(train_loader):
        current_step += 1
        if current_step > total_iters:
            break
        # updating learning rate
        model.update_learning_rate()

        # training
        model.feed_data(train_data)
        model.optimize_parameters()

        # log
        if current_step % config['log_iter'] == 0:
            logs = model.get_current_log()
            message = '[epoch:{:3d}, iter:{:8,d}, lr:{:.3e}] '.format(epoch, current_step,
                                                                      model.get_current_learning_rate())
            for k, v in logs.items():
                message += '{:s}: {:.4f} '.format(k, v)
                # tensorboard logger
                train_writer.add_scalar(k, v, current_step)
            print(message)

        # validation
        if current_step % config['val_iter'] == 0:
            v_input, v_output, v_target = [], [], []
            visual_images = []
            for index, val_data in enumerate(val_loader):
                if index < config['display_num']:
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    v_input.append(visuals['input'])
                    v_output.append(visuals['output'])
                    v_target.append(visuals['target'])
                else:
                    break

            visual_images.extend(v_input)
            visual_images.extend(v_output)
            visual_images.extend(v_target)
            _write_images(visual_images, config['display_num'], '%s/val_current.jpg' % image_dir)

        # save images and html file
        if current_step % config['save_image_iter'] == 0:
            v_input, v_output, v_target = [], [], []
            visual_images = []
            for index, val_data in enumerate(val_loader):
                if index < config['display_num']:
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    v_input.append(visuals['input'])
                    v_output.append(visuals['output'])
                    v_target.append(visuals['target'])
                else:
                    break

            visual_images.extend(v_input)
            visual_images.extend(v_output)
            visual_images.extend(v_target)
            _write_images(visual_images, config['display_num'], '%s/val_%08d.jpg' % (image_dir, current_step))
            # HTML
            write_html(output_dir + '/index.html', current_step, config['save_image_iter'], 'images')

        # save models
        if current_step % config['save_model_iter'] == 0:
            print('Saving models.')
            model.save(current_step)

print('Saving the final model.')
model.save('latest')
print('End of training.')

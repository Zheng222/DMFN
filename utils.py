import yaml
import torch
import torchvision.utils as vutils
import os
import numpy as np

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def _write_images(images, display_image_num, file_name):  # images is a list that contains tensors with shape [N,C,H,W]
    image_tensor = torch.stack(images, dim=0)
    image_grid = vutils.make_grid(image_tensor, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations, img_filename.split('/')[-1]))
    html_file.write("""
        <p>
            <a href="%s">
                <img scr="%s" style="width:%dpx">
            </a></br>
        </p> 
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment name = %s</title>
        <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/val_current.jpg' % (image_directory), all_size)
    for i in range(iterations, image_save_iterations - 1, -1):
        if i % image_save_iterations == 0:
            write_one_row_html(html_file, i, '%s/val_%08d.jpg' % (image_directory, i), all_size)
    html_file.write("</body></html>")
    html_file.close()


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max, 1)) / x_max * 2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max, 1)) / y_max * 2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor

def batch_get_centers(fea, epsilon=1e-7):
    B, C, H, W = fea.shape
    fea_map = fea + epsilon
    k = torch.sum(fea_map, dim=(2, 3), keepdim=True)  # reduce dim = H, W
    fea_map_pdf = fea_map / k

    x_map, y_map = get_coordinate_tensors(H, W)
    x_center = torch.sum(fea_map_pdf * x_map.repeat(B, C, 1, 1), dim=(2, 3), keepdim=True)
    y_center = torch.sum(fea_map_pdf * y_map.repeat(B, C, 1, 1), dim=(2, 3), keepdim=True)

    return torch.cat((x_center, y_center), dim=3)  # B, C, 1, 2

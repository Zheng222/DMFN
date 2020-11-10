import models.architecture as arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'DMFN':
        netG = arch.InpaintingGenerator(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                        n_res=opt_net['n_res'])
    else:
        raise NotImplementedError('Unsupported generator model: {}'.format(which_model))

    return netG
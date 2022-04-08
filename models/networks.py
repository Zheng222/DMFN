import models.architecture as arch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'DMFN':
        netG = arch.InpaintingGenerator(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                        n_res=opt_net['n_res'])
    else:
        raise NotImplementedError('Unsupported generator model: {}'.format(which_model))
    if opt['is_train']:
        netG.apply(weights_init)
    return netG

# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator':
        netD = arch.Discriminator(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Unsupported discriminator model: {}'.format(which_model))

    netD.apply(weights_init)
    return netD



def define_F(opt):
    netF = arch.VGGFeatureExtractor()
    netF.eval()
    return netF

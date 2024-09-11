
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import utils.util as util
import torch.nn.utils.spectral_norm as spectral_norm
import argparse

def get_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type = norm_type

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        # elif subnorm_type == 'sync_batch':
        #     norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError(
                'normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

class MultiscaleDiscriminator(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

        # self.init_weights(opt.init_type, opt.init_variance)

    def create_single_discriminator(self, opt):
        netD = NLayerDiscriminator(opt)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = self.opt.fm_power > 0
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = opt.input_channel

        norm_layer = get_norm_layer(opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, opt.max_conv_dim)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[
                norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                           stride=stride, padding=padw)),
                nn.LeakyReLU(0.2, False)
            ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        get_intermediate_features = self.opt.fm_power > 0
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]




def get_parser():
    parser = argparse.ArgumentParser(description='Options for training SwapYou')

    ###1. dataset
    parser.add_argument('--dataroot', type=str, default='data/ZHEN_cleaned_rig_train.pkl')
    parser.add_argument('--val_dataroot', type=str, default='data/ZHEN_cleaned_rig_train.pkl')
    ###2. model arch
    parser.add_argument('--input_channel', type=int, default=3, help='# of input image channels')
    parser.add_argument('--input_size', type=int, default=256, help='# of input image w&h')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--mask_df', type=int, default=16, help='# of discrim filters in first conv layer')

    parser.add_argument('--data_view', type=str, default=None)
    parser.add_argument('--n_layers_D', type=int, default=4, help='num of layers of D')
    parser.add_argument('--num_D', type=int, default=3, help='# of discrim')
    parser.add_argument('--type_D', type=str, default='patch', help='type of discrim')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--sn_G', action='store_true', help='spectral norm for generator')
    parser.add_argument('--sn_D', action='store_true', help='spectral norm for discriminator')
    parser.add_argument('--epoch', type=int, default=20, help='# total epochs')
    parser.add_argument('--decay_epoch', type=int, default=10, help='#epoch to decay learning rate')
    parser.add_argument('--decay_factor', type=float, default=0.9, help='#decay_factor to decay learning rate')
    parser.add_argument('--id_th', type=float, default=0.0)
    parser.add_argument('--id_power', type=float, default=-1.0)
    parser.add_argument('--vgg_power', type=float, default=0.0)
    parser.add_argument('--ssim_power', type=float, default=0.0)
    parser.add_argument('--mouth_power', type=float, default=0.0)
    parser.add_argument('--eyes_power', type=float, default=0.0)
    parser.add_argument('--sobel_power', type=float, default=0.0)
    parser.add_argument('--face_power', type=float, default=0.0)

    parser.add_argument('--id_ch', type=int, default=1024, help="dimension of the id")
    parser.add_argument('--content_ch', type=int, default=8, help="dimension of the id")
    parser.add_argument('--background_ch', type=int, default=1, help="dimension of the id")

    parser.add_argument('--bottleneck_nums', type=int, default=2, help='bottleneck_nums size')
    parser.add_argument('--max_conv_dim', type=int, default=512, help='max_conv_dim')
    parser.add_argument('--bottle_dim', type=int, default=16, help='bottle_dim')
    parser.add_argument('--exp_dim', type=int, default=16, help='bottle_dim')

    parser.add_argument('--id_num', type=int, default=4, help='the number of id')

    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    parser.add_argument('--num_upsampling_layers', default=5,type=int,
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

    parser.add_argument('--skip_module', action='store_true', help='using skip_module')
    parser.add_argument('--skip_module_list', nargs='*', type=int, help="skip_module_list for details")
    parser.add_argument('--no_adain_list', nargs='*', type=int, help="skip_module_list for details")
    parser.add_argument('--attention_type', type=str, default='spatial', help='(spatial|channel|both)')

    parser.add_argument('--new_D', action='store_true', help='using multi domain discriminator')
    parser.add_argument('--G_v2', action='store_true', help='using multi domain discriminator')


    parser.add_argument('--exp_injection', action='store_true', help='inject expression embedding')
    parser.add_argument('--residual_learning', action='store_true', help='inject expression embedding')
    parser.add_argument('--exp_reduce', action='store_true', help='inject expression embedding')
    parser.add_argument('--merge_norm', type=str, default='batchnorm', help='norm type of merge module')

    parser.add_argument('--exp_injection_type', type=str, default='adain', help='inject expression embedding')


    parser.add_argument('--pretrained_model', type=str, default=None, help='inject expression embedding')

    parser.add_argument('--id_injection_models', type=str, default='facenet#arcfacev2')
    parser.add_argument('--id_constrain_models', type=str, default='facenet#5+arcfacev2#10')
    parser.add_argument('--id_fusion_method', type=str, default='concat', help='(concat|add)')
    ###3. log
    parser.add_argument('--name', type=str, default='SwapYou', help='name of the experiment.')
    parser.add_argument('--arch', type=str, default='dcgan', help='name of the experiment.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_wild', help='models are saved here')
    parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
    parser.add_argument('--save_step_freq', type=int, default=20000, help='frequency of steps to save model')
    parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of showing training results on console')

    parser.add_argument('--resume_dir', type=str, default=None , help='models are saved here')

    ###4. pretrained checkpints
    parser.add_argument('--arcfacev2_model_path', type=str, default='/data/input_models/arcfacev2_resnet50_IR_checkpoint.pth')
    parser.add_argument('--facenet_model_path', type=str, default='/data/input_models/20180402-114759-vggface2.pt')
    parser.add_argument('--cosface_model_path', type=str, default='/data/input_models/cosface_ACC99.28.pth')
    parser.add_argument('--arcface_irse_model_path', type=str, default='/data/input_models/model_ir_se50.pth')
    parser.add_argument('--arcface_insight_model_path', type=str, default='/data/input_models/Glint360k_r100_backbone.pth')
    parser.add_argument('--exp_model_path', type=str, default='/data/input_models/minus_pipeline_DISFA_FEC_best_aug_857.pth')

    ###5. loss
    parser.add_argument('--cycle_power', type=float, default=0.0, help='weight for cycle loss')
    parser.add_argument('--cycle_start_epoch', type=int, default=1)
    parser.add_argument('--exp_power', type=float, default=10.0, help='weight for exp loss')
    parser.add_argument('--neg_exp_power', type=float, default=0.0, help='weight for neg_exp loss')
    parser.add_argument('--gan_power', type=float, default=1.0, help='weight for gan loss')
    parser.add_argument('--neg_id_power', type=float, default=0.0, help='weight for neg_id loss')
    parser.add_argument('--neg_id_exp_power', type=float, default=0.0, help='weight for neg_id loss')
    parser.add_argument('--rec_power', type=float, default=10.0, help='weight for rec loss')
    parser.add_argument('--fm_power', type=float, default=0.0, help='weight for fm loss')
    parser.add_argument('--tv_power', type=float, default=0.0, help='weight for total_variation_loss')
    parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
    parser.add_argument('--mask_power', type=float, default=10.0, help='weight for mask loss')
    parser.add_argument('--rig_power', type=float, default=10.0, help='weight for mask loss')
    parser.add_argument('--symmetry_power', type=float, default=0.0, help='weight for mask loss')

    ###5. training parameters
    parser.add_argument('--seed', type=int, default=666666, help='seed')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='momentum term of adam')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='initial learning rate for G')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='initial learning rate for D')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--same_prob', type=float, default=0.0, help='same prob')

    args = parser.parse_args()

    return args

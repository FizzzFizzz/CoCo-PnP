import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import logging
import os
import sys 

sys.path.append("..") 
import utils_image as util
import utils_logger
from network_unet import UNetRes as Net


    



model_path = 'CoCo4_color.pth'
n_channels = 3



model = Net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose", bias=False)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()
device = 'cuda'
for k, v in model.named_parameters():
    v.requires_grad = False




class Drunet_running(torch.nn.Module):
    def __init__(self):
        super(Drunet_running, self).__init__()
        # self.models = {}
        # for level in models:
        #     self.models[level] = models[level]
        #     self.models[level].eval()
        self.models = model
        self.models.eval()
    
    def to(self, device):
        
        self.models.to(device)    

    def forward(self, x, sigma):
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        x = x.to(device)
        sigma = float(sigma)
        # sigma_div_255 = torch.FloatTensor([sigma/255.]).repeat(1, 1, x.shape[2], x.shape[3]).cuda()
        sigma_div_255 = torch.FloatTensor([sigma/255.]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)
        x = torch.cat((x, sigma_div_255), dim=1)
        return self.models(x)



def run_model(x, sigma):       
    '''
        x is image in [0, 1]
        simga in [0, 255]
    '''
    # print(x.size())
    sigma = float(sigma)
    sigma_div_255 = torch.FloatTensor([sigma]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)
    # sigma_div_255 = 0*x + sigma
    x = torch.cat((x, sigma_div_255), dim=1)

    return model(x)
# # #









def print_line(y, pth, label):
    x = range(len(y))
    plt.plot(x, y, '-', alpha=0.8, linewidth=1.5, label=label)
    plt.legend(loc="upper right")
    plt.xlabel('iter')
    plt.ylabel(label)
    plt.savefig(pth)
    plt.close()    

# nb: default 30 for CBSD68
class PnP_ADMM(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nb=10, act_mode='R'):
        super(PnP_ADMM, self).__init__()
        self.nb = nb

        self.net = Drunet_running()
        # self.net = run_model()

        # only test
        self.res = {}
        self.res['psnr'] = [0] * nb
        self.res['ssim'] = [0] * nb
        self.res['image'] = [0]* nb
        self.res['mse'] = [0]*nb

    def get_psnr_i(self, u, clean, i):
        pre_i = torch.clamp(u / 255., 0., 1.)
        img_E = util.tensor2uint(pre_i)
        img_H = util.tensor2uint(clean)
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        # print(psnr)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        
        self.res['psnr'][i] = psnr
        self.res['ssim'][i] = ssim
        #if i > 0:
        #    if self.res['psnr'][i] == max(self.res['psnr']):
        self.res['image'][i] = ToPILImage()(pre_i[0])
        

    def forward(self, initial_uv, f, clean, sigma=25.5, lamb=690, sigma2=1.0, denoisor_sigma=25, irl1_iter_num=10, eps=1e-5,alpha = 0.): 
        # init
        f *= 255
        u  = initial_uv * 255
        w = u
        v  = initial_uv * 255
        # b2 = torch.zeros(f.shape, device='cpu')

        z = u
        w = u
        lamb_ = lamb
        # ADMM
        

        for k in range(self.nb):

            oldw = w
            oldz = z

            if k==0:
                self.get_psnr_i(w/torch.max(w)*255, clean, k)
            else:
                self.get_psnr_i(w, clean, k)
            # self.get_psnr_i(w, clean, k)
            # exact fidelity

            # print(k)
            u = z-lamb_+((z-lamb_)**2+4*lamb_*f)**0.5
            u = u / 2
            v= 2*u-z
            model_input = (v) / 255.   
            # model_input = torch.clamp(model_input, min = -1., max =2.)           
            w=run_model(model_input,denoisor_sigma) * 255.
            k0 = 0.33
            tw = k0 * w + (1-k0)*v
            z = 2*tw-v
            z = 0.5*z + 0.5*oldz
            


            newnew = util.tensor2uint(w/255)
            oldold = util.tensor2uint(oldw/255)

            mse_ = np.mean((newnew-oldold)**2)
            self.res['mse'][k] = mse_
            
            
        # return u # ADMM or DRS
        return w # 

def plot_psnr(denoisor_level, lamb, sigma):
    device = 'cuda'
    model = PnP_ADMM()
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()

    sigma2 = 1.0

    fp = 'CBSD68_cut8/0020.png'
    img_H = util.imread_uint(fp, 3)
    print(img_H.shape)
    initial_uv, img_L, img_H = gen_data(img_H, sigma)

    print(img_H.shape)
    print(img_L.shape)
    print(initial_uv.shape)

    initial_uv = initial_uv.to(device)
    img_L = img_L.to(device)
    img_H = img_H.to(device)




    with torch.no_grad():
        img_L, img_H = img_L.to(device), img_H.to(device)
        # model(img_L, img_H, sigma, lamb, sigma2, denoisor_level, 10, 1e-5)
        model(initial_uv, img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5)

    savepth = 'images/'
    for j in range(len(model.res['image'])):
        # model.res['image'][j].save(savepth + 'result_Brain{}_{}.png'.format(i, j))
        model.res['image'][j].save(savepth + 'result_{}.png'.format(j))

    y = model.res['psnr']
    print(y)
    print(y[-1])
    print(np.max(y))
    x = range(len(y))
    plt.figure()
    plt.plot(x, y, '-', alpha=0.8, linewidth=1.5)
    plt.xlabel('iter')
    plt.ylabel('PSNR')
    plt.savefig('PSNR_level{}_lamb{}.png'.format(denoisor_level, lamb))

    # plt.figure()
    # plt.plot(x, model.res['mse'], '-', alpha=0.8, linewidth=1.5)
    # plt.xlabel('iter')
    # plt.ylabel('MSE')
    # plt.savefig('MSE_level{}_lamb{}.png'.format(denoisor_level, lamb))






def gen_data(img_clean_uint8, sigma):
    img_H = img_clean_uint8
    img_H = util.uint2single(img_H)
    img_L = np.copy(img_H)

    np.random.seed(seed=0)


    img_L = np.float32(np.random.poisson(img_L * sigma) / sigma)


    img_L = util.single2tensor3(img_L).unsqueeze(0)
    img_H = util.single2tensor3(img_H).unsqueeze(0)
    # initial_uv = MedianFilter(img_L, 3, None)
    initial_uv = img_L
    return initial_uv, img_L, img_H






def search_args():
    sigma = 30
    utils_logger.logger_info('poisson', log_path='log/peak_{}/logger.log'.format(sigma))
    logger = logging.getLogger('poisson')
    device = 'cuda'

    logger.info('peak = {}'.format(sigma))

    model = PnP_ADMM()
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()


    dataset_root = 'CBSD68_cut8/'

    max_psnr   = -1
    max_level  = -1
    max_lamb   = -1
    max_sigma2 = -1
    max_ssim   = -1

    search_range = {}




    alpha = 0.




    # # k0 = 0.33. Coco4, alpha = 0
    # ## peak = sigma = 30, CBSD68, poisson noise removal task.
    # search_range[0.10] = [73] # 29.9947, 0.8604
    # ## peak = sigma = 20, CBSD68, poisson noise removal task.
    # search_range[0.12] = [67] # 28.7592, 0.8323
    # ## peak = sigma = 15, CBSD68, poisson noise removal task.
    # search_range[0.15] = [85] # 27.8000, 0.8068
    # ## peak = sigma = 10, CBSD68, poisson noise removal task.
    # search_range[0.18] = [70,75,80] # 

    # ## peak = sigma = 30, DIV2K Val, poisson noise removal task.
    search_range[0.10] = [70,73,76] # 
    # ## peak = sigma = 20, DIV2K Val, poisson noise removal task.
    # search_range[0.12] = [67] # 




    search_level = [0.10]# 

    psnr_save_root  = 'log/' + 'peak_' + str(sigma) + '/psnr'
    ssim_save_root  = 'log/' + 'peak_' + str(sigma) + '/ssim'
    image_save_root = 'log/' + 'peak_' + str(sigma) + '/image'
    if not os.path.exists(psnr_save_root):
        os.makedirs(psnr_save_root)    
    if not os.path.exists(ssim_save_root):
        os.makedirs(ssim_save_root)   
    if not os.path.exists(image_save_root):
        os.makedirs(image_save_root)

    for denoisor_level in search_level:
        logger.info('========================================')
        logger.info('denoisor_level: {}'.format(denoisor_level))
        logger.info('alpha: {}'.format(alpha))
        logger.info('========================================')
        for sigma2 in [1.]: 
            # for lamb in range(*search_range[denoisor_level]):
            for lamb in search_range[denoisor_level]:
                logger.info('==================')
                logger.info('lamb: {}'.format(lamb))

                dataset_psnr = None
                dataset_ssim = None
                image_paths = util.get_image_paths(dataset_root)
                # noise_paths = util.get_image_paths(dataset_noise_root)
                # med_paths   = util.get_image_paths(dataset_med_root)
                # image_paths = [image_paths[i] for i in range(0,181,10)]
                # for fp in tqdm(image_paths):
                image_number=12
                image_number = len(image_paths)
                for ii in range(0,image_number):
                    fp = image_paths[ii]
                    img_H = util.imread_uint(fp, 3)
                    
                    initial_uv, img_L, img_H = gen_data(img_H, sigma)
                    initial_uv = initial_uv.to(device)
                    img_L = img_L.to(device)
                    img_H = img_H.to(device)

                    with torch.no_grad():
                        model(initial_uv, img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5, alpha)

                    cur_psnr = np.array(model.res['psnr'])
                    # print(ii+1)
                    # print(cur_psnr[-1]) # the last i-th image's psnr value.
                    print(np.max(cur_psnr)) # the highest psnr value.
                    cur_ssim = np.array(model.res['ssim'])
                    if dataset_psnr is None:
                        dataset_psnr = cur_psnr
                        dataset_ssim = cur_ssim
                    else:
                        dataset_psnr += cur_psnr
                        dataset_ssim += cur_ssim
                # dataset_psnr /= len(image_paths)
                # dataset_ssim /= len(image_paths)
                dataset_psnr /= image_number
                dataset_ssim /= image_number
                print(dataset_psnr.shape)


                # # # choose as you wish :)
                # cur_avg_psnr = np.max(dataset_psnr)
                # cur_avg_ssim = np.max(dataset_ssim)
                cur_avg_psnr = dataset_psnr[-1]
                cur_avg_ssim = dataset_ssim[-1]

                logger.info("PSNR: {:.4f}".format(cur_avg_psnr))
                logger.info("SSIM: {:.4f}".format(cur_avg_ssim))
                psnr_save_pth = psnr_save_root + '/level' + str(denoisor_level) + '_lamb' + str(lamb) + '_psnr' + str(cur_avg_psnr)[:7] + '.png'
                ssim_save_pth = ssim_save_root + '/level' + str(denoisor_level) + '_lamb' + str(lamb) + '_psnr' + str(cur_avg_psnr)[:7] + '.png'
                # image_save_pth = image_save_root + '/level' + str(denoisor_level) + '_lamb' + str(lamb) + '_psnr' + str(cur_psnr)[:7] + '.png'
                print_line(dataset_psnr, psnr_save_pth, "PSNR")
                print_line(dataset_ssim, ssim_save_pth, "SSIM")

                if cur_avg_psnr > max_psnr:
                    max_psnr   = cur_avg_psnr
                    max_level  = denoisor_level
                    max_lamb   = lamb
                    max_ssim   = cur_avg_ssim
                    # max_sigma2 = sigma2
                    # print(model.res['l'])


    logger.info('========================================')
    logger.info('========================================')
    logger.info('max_psnr: {:4f}'.format(max_psnr))
    logger.info('max_ssim: {:4f}'.format(max_ssim))
    logger.info('level: {}'.format(max_level))
    logger.info('lamb: {}'.format(max_lamb))
    return max_psnr, max_level, max_lamb






# note that sigma is the peak in the poisson noise removal settings.

# max_psnr, max_level, max_lamb = search_args()

# 0029.png, peak=
# plot_psnr(0.18, 75, 10) # 26.51
# plot_psnr(0.15, 80, 15) # 27.95
# plot_psnr(0.12, 67, 20) # 28.86
# plot_psnr(0.10, 70, 30) # 30.03



# 0000.png, peak=

# plot_psnr(0.12, 20, 20) # 36.47/37.88
# plot_psnr(0.08, 20, 20) # 36.88/36.88


# 0020.png, peak=
# plot_psnr(0.18, 100, 10) # 
# plot_psnr(0.14, 140, 20) # 30.29
# plot_psnr(0.12, 70, 20) # 30.23
# plot_psnr(0.10, 40, 20) # 30.15


# 0891.png,
plot_psnr(0.12, 70, 20) # 30.23


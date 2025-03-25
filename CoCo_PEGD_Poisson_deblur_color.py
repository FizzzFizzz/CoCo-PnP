import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import logging
import os
import sys 
from PIL import Image
sys.path.append("..") 


import utils_image as util
import utils_logger

    
from network_unet import UNetRes as Net
import utils_deblur as deblur

    
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

# nb: default 100 for BSD68
class PnP_PEGD(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nb=100, act_mode='R'):
        super(PnP_PEGD, self).__init__()
        self.nb = nb

        self.net = Drunet_running()
        # self.net = run_model()

        # only test
        self.res = {}
        self.res['psnr'] = [0] * nb
        self.res['ssim'] = [0] * nb
        self.res['image'] = [0]* nb
        self.res['mse'] = [0]*nb

    def IRL1(self, f, u, v, b2, sigma, lamb, sigma2, k=10, eps=1e-5):
        for j in range(k):
            fenzi = lamb * (v-f)/(sigma**2+(v-f)**2)+(v-u-b2)
            fenmu = lamb * (sigma**2-(v-f)**2)/(sigma**2+(v-f)**2)**2+1
            v = v - fenzi / fenmu
            v = torch.clamp(v, min=0, max=255.)



        return v

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
        

    def forward(self, kernel, initial_uv, f, clean, sigma=25.5, lamb=690, sigma2=1.0, denoisor_sigma=25, irl1_iter_num=10, eps=1e-5, dt = 0.): 
        # init
        f *= 255
        u  = f
        v  = f
        w  = f

        oldu = u

        w = u
        x = u

        bx = 0*u
        tw = u



        K = kernel
        fft_k = deblur.p2o(K, u.shape[-2:])
        fft_kH = torch.conj(fft_k)
        abs_k = fft_kH * fft_k
        
        rho = 1

        fenmu2 = rho*abs_k+1
        for k in range(self.nb):
            if k==0:
                self.get_psnr_i(w/torch.max(w)*255, clean, k)
            else:
                self.get_psnr_i(w, clean, k)
            
            oldu = u



            model_input = (v) / 255.   
            model_input = model_input.type(torch.cuda.FloatTensor)        
            w=run_model(model_input,denoisor_sigma) * 255.
            k0 = 0.33
            tw = k0 * w + (1-k0)*(v)
            u = tw
            


            # This inner iteration solves the subproblem: argmin_v { lamb <1, Kv-f logKv> + 1/2 \|v - (b+u) \|^2 } 
            lamb_ = lamb
            bx = 0*x
            vv = u
            x = torch.real(deblur.ifftn(fft_k*deblur.fftn(vv)))
            for _ in range(10):
                tmp = torch.real(deblur.ifftn(fft_k*deblur.fftn(vv)))
                input_x = tmp-bx
                x = (input_x-lamb_+((input_x-lamb_)**2+4*lamb_*f)**0.5) / 2
                fenzi2 = u + torch.real(deblur.ifftn(fft_kH*deblur.fftn(x+bx)))
                vv = torch.real(deblur.ifftn(deblur.fftn(fenzi2)/fenmu2))
                bx = bx + x - tmp

            # dt = lamb_/2

            v = (1-dt) * u + dt* vv

            

            

            
            

            newnew = util.tensor2uint(u/255)
            oldold = util.tensor2uint(oldu/255)

            mse_ = np.mean((newnew-oldold)**2)
            self.res['mse'][k] = mse_
            
            
        return w # 

def plot_psnr(denoisor_level, lamb, sigma, dt):
    device = 'cuda'
    model = PnP_PEGD()
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()

    sigma2 = 1.0


    

    #这里是在读图和kernel。
    kernel_fp = 'kernels/levin_8.png'

    fp = 'CBSD68_cut8/0005.png'

    kernel = util.imread_uint(kernel_fp,1) #这个1是指读灰度图。
    kernel = util.single2tensor3(kernel).unsqueeze(0) / 255.
    kernel = kernel / torch.sum(kernel)
    img_H = util.imread_uint(fp, 3)
    img_H = util.single2tensor3(img_H).unsqueeze(0) /255.
    print(img_H.shape)
    initial_uv, img_L, img_H = gen_data(img_H, sigma, kernel)

    print(img_H.shape)
    print(img_L.shape)
    print(initial_uv.shape)

    # initial_uv = initial_uv.to(device)
    img_L = img_L.to(device)
    img_H = img_H.to(device)
    kernel = kernel.to(device)




    with torch.no_grad():
        img_L, img_H= img_L.to(device), img_H.to(device)
        # model(img_L, img_H, sigma, lamb, sigma2, denoisor_level, 10, 1e-5)
        model(kernel, initial_uv, img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5,dt)
    savepth = 'images/'
    for j in range(len(model.res['image'])):
        # model.res['image'][j].save(savepth + 'result_Brain{}_{}.png'.format(i, j))
        model.res['image'][j].save(savepth + 'result_{}.png'.format(j))

    y = model.res['psnr']
    # print(y)
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






def gen_data(img_clean_uint8, sigma,kernel):

    img_H = img_clean_uint8
    img_L = img_clean_uint8
    
    fft_k = deblur.p2o(kernel, img_L.shape[-2:])

    temp = fft_k * deblur.fftn(img_L)
    img_L = torch.real(deblur.ifftn(temp))
    img_L = torch.clamp(img_L, 0., 10.)
    np.random.seed(seed=0)
    img_L = np.float32(np.random.poisson(img_L * sigma) / sigma)
    img_L = torch.from_numpy(img_L)
    initial_uv = img_L
    # print(img_L.type())
    # print(img_H.type())
    return initial_uv, img_L, img_H





def search_args():
    sigma = 100
    utils_logger.logger_info('deblur', log_path='log/peak_{}/logger.log'.format(sigma))
    logger = logging.getLogger('deblur')
    device = 'cuda'

    logger.info('peak = {}'.format(sigma))

    model = PnP_PEGD()
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

    kernel_fp = 'kernels/levin_8.png'
    ## peak =  100, CBSD68_cut, poisson noise removal task. Coco with r=0.25, t=0.33
    # kernel 1
    search_range[0.1] = [460] # 26.4893, 0.7161
    # kernel 2
    # search_range[0.1] = [500] # 26.3330, 0.7119
    # kernel 3
    # search_range[0.1] = [560] # 26.8554, 0.7343
    # kernel 4
    # search_range[0.1] = [520] # 26.1000, 0.7028
    # kernel 5
    # search_range[0.1] = [520] # 27.7006, 0.7693
    # kernel 6
    # search_range[0.1] = [480] # 27.4020, 0.7570
    # kernel 7
    # search_range[0.1] = [560] # 26.9173, 0.7421
    # kernel 8
    # search_range[0.1] = [540] # 26.5067, 0.7246


    
    ## peak =  50, CBSD68_cut, poisson noise removal task. Coco with r=0.25, t=0.33
    # kernel 1
    # search_range[0.12] = [340] # 25.5948, 0.6801
    # kernel 2
    # search_range[0.12] = [360] # 25.4708, 0.6762
    # kernel 3
    # search_range[0.12] = [360] # 26.0688, 0.6994
    # kernel 4
    # search_range[0.12] = [360] # 25.2271, 0.6654
    # kernel 5
    # search_range[0.12] = [360] # 26.7807, 0.7354
    # kernel 6
    # search_range[0.12] = [320] # 26.4150, 0.7170
    # kernel 7
    # search_range[0.12] = [360] # 26.0214, 0.7059
    # kernel 8
    # search_range[0.12] = [360] # 25.5852, 0.6869




    dt = 0.8
    

    search_level = [0.1]# [5, 10, 15, 20, 25, 30]

    


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
        logger.info('dt: {}'.format(dt))
        logger.info('========================================')
        for sigma2 in [1.]: 
            # for lamb in range(*search_range[denoisor_level]):
            for lamb in search_range[denoisor_level]:
                logger.info('==================')
                logger.info('lamb: {}'.format(lamb))

                dataset_psnr = None
                dataset_ssim = None
                image_paths = util.get_image_paths(dataset_root)

                image_number = len(image_paths)
                print(image_number)
                for ii in range(0,image_number):
                    fp = image_paths[ii]
                    img_H = util.imread_uint(fp, 3)
                    img_H = util.single2tensor3(img_H).unsqueeze(0) /255.
                    kernel = util.imread_uint(kernel_fp,1) #这个1是指读灰度图。
                    kernel = util.single2tensor3(kernel).unsqueeze(0) / 255.
                    kernel = kernel / torch.sum(kernel)
                    initial_uv, img_L, img_H = gen_data(img_H, sigma, kernel)
                    img_L = img_L.to(device)
                    img_H = img_H.to(device)
                    kernel = kernel.to(device)

                    with torch.no_grad():
                        img_L, img_H= img_L.to(device), img_H.to(device)
                        model(kernel, initial_uv,img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5, dt)

                    cur_psnr = np.array(model.res['psnr'])
                    # print(cur_psnr)
                    # print(ii+1)
                    print(cur_psnr[-1]) # the last i-th image's psnr value.
                    # print(np.max(cur_psnr)) # the highest psnr value.
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


# plot_psnr(level, lambda, peak, dt), 
# lambda controls the envelope ME_f(input;lamb)=min_x f(x) + 1/(2*lamb) \|x-input\|^2. f(x) = <1, Kx-flog(Kx)>
# Smaller lamb means more accurate approximation
# dt controls the step size on the gradient of ME_f.

# max_psnr, max_level, max_lamb = search_args()



# 0005.png, kernel8, 25.33 by Coco4-ADMM
plot_psnr(0.12, 360, 50, 0.8) # 25.33, k0=0.33
# plot_psnr(0.1, 500, 100, 0.8) # 26.39, k0=0.33

# butterfly, kernel2, 25.25 by Coco4-ADMM
# plot_psnr(0.12, 300, 50, 0.8) # 25.17, k0=0.33

# butterfly, kernel2, 25.25 by Coco4-ADMM
# plot_psnr(0.12, 300, 50, 0.8) # 25.17, k0=0.33

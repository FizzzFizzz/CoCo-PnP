import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import logging
import os
import sys 
import cv2
sys.path.append("..") 


import utils_image as util
import utils_logger
from utils import utils_sisr as sr_old
import utils_sisr_wdl as sr


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

        self.models = model
        self.models.eval()
    
    def to(self, device):
        
        self.models.to(device)    

    def forward(self, x, sigma):
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        x = x.to(device)
        sigma = float(sigma)

        sigma_div_255 = torch.FloatTensor([sigma/255.]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)
        x = torch.cat((x, sigma_div_255), dim=1)
        return self.models(x)



def run_model(x, sigma):       
    '''
        x is image in [0, 1]
        simga in [0, 255]
    '''

    sigma = float(sigma)
    sigma_div_255 = torch.FloatTensor([sigma]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)

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

# nb: default 50.
class PnP_ADMM(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nb=101, act_mode='R'):
        super(PnP_ADMM, self).__init__()
        self.nb = nb

        self.net = Drunet_running()
        # self.net = run_model()

        # only test
        self.res = {}
        self.res['psnr'] = [0] * nb
        self.res['ssim'] = [0] * nb
        self.res['image'] = [0]* nb



    def get_psnr_i(self, u, clean, i):
        pre_i = torch.clamp(u / 255., 0., 1.)
        img_E = util.tensor2uint(pre_i)
        img_H = util.tensor2uint(clean)
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)

        self.res['psnr'][i] = psnr
        self.res['ssim'][i] = ssim

        self.res['image'][i] = ToPILImage()(pre_i[0])

    def forward(self, kernel, initial_uv, f, clean, sigma=25.5, lamb=690, sigma2=1.0, denoisor_sigma=25, irl1_iter_num=10, eps=1e-5,sf=4): 

        f *= 255

        ## intialization of x and y
        x = f.clone()*0
        y = f.clone()*0

        temp = f.squeeze(0)
        temp = temp.squeeze(0)
        temp = temp.unsqueeze(-1)
        temp = temp.cpu().numpy()
        # print(temp.shape)
        yy = np.zeros([1,3,clean.shape[2],clean.shape[3]])
        for iiii in range(3):
            x = np.zeros([temp.shape[1],temp.shape[2]])
            x[:,:] = temp[iiii,:,:,0]
            x = cv2.resize(x, (x.shape[1]*sf, x.shape[0]*sf), interpolation=cv2.INTER_CUBIC)
            x = sr_old.shift_pixel(x, sf)
            yy[0,iiii,:,:] = x[:,:]

        yy = torch.from_numpy(yy)
        ## intialization of u
        u = yy.type(torch.cuda.FloatTensor)

        K = kernel

        ## intialization of v and b
        v = u.clone()
        b = u.clone()*0





        lamb_ = lamb
        d = denoisor_sigma
        t = u
        
        # img_L_tensor = f / 255
        # k_tensor = K
        # FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, sf)



        for k in range(self.nb):
            self.get_psnr_i(torch.clamp(t, min = -0., max =255.), clean, k)

            step = 0.33
            # step = 1.0
            
            rho = 1.
            for _ in range(1):
                _, Sku, _ = gen_data((u/255).cpu(), 0, K.cpu(), sf)
                Sku = (Sku*255).to(device)
                input = Sku - y
                x = (input-lamb_/rho+((input-lamb_/rho)**2+4*lamb_/rho*f)**0.5) / 2

                FB_, FBC_, F2B_, FBFy_ = sr.pre_calculate(((x+y)/255).float(), K, sf)
                u = sr.data_solution(((v-b)/255).float(), FB_, FBC_, F2B_, FBFy_, 1/rho, sf)*255
                
                _, Sku, _ = gen_data((u/255).cpu(), 0, K.cpu(), sf)
                Sku = (Sku*255).to(device)
                y = y + x - Sku

            x = x.type(torch.cuda.FloatTensor)
            u = u.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)

            t = run_model((u+b)/255,d) * 255
            v = step*t+(1-step)*(u+b)

            b = b + u - v

            


            # if step == 1.0:
            #     ratio = 1.1273
            #     lamb_ = lamb_ / ratio
            #     d = d / (ratio)**0.5
            

        return torch.clamp(t, min = -0., max =255.) # HBS/HQS


def create_gaussian_kernel(size, sigma):
    # 创建一个大小为size x size的二维数组
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    g = torch.tensor(g, dtype=torch.float32)
    return g / g.sum() 

# gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
# gaussian_kernel = torch.tensor(gaussian_kernel, dtype=torch.float32)

def plot_psnr(denoisor_level, lamb, sigma, sf):
    
    device = 'cuda'
    model = PnP_ADMM()
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()
    
    sigma2 = 1.0
    fp = 'CBSD68_cut8/0046.png'



    # create_gaussian_kernel(kernel_size, sigma)
    kernel = create_gaussian_kernel(11, 1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # print(kernel.shape)
    img_H = util.imread_uint(fp, 3)
    img_H = util.single2tensor3(img_H).unsqueeze(0) /255.
    initial_uv, img_L, img_H = gen_data(img_H, sigma,kernel, sf)
    

    initial_uv = initial_uv.to(device)
    img_L = img_L.to(device)
    img_H = img_H.to(device)
    kernel = kernel.to(device)




    with torch.no_grad():
        img_L, img_H = img_L.to(device), img_H.to(device)
        kernel = kernel.to(device)

        model(kernel, initial_uv, img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5,sf)

    savepth = 'images/'
    for j in range(len(model.res['image'])):
        # model.res['image'][j].save(savepth + 'result_Brain{}_{}.png'.format(i, j))
        model.res['image'][j].save(savepth + 'result_{}.png'.format(j))

    y = model.res['psnr']
    # print(y)
    print(y[-1])
    x = range(len(y))
    plt.plot(x, y, '-', alpha=0.8, linewidth=1.5)

    plt.xlabel('iter')
    plt.ylabel('PSNR')

    plt.savefig('PSNR_level{}_lamb{}.png'.format(denoisor_level, lamb))






def gen_data(img_clean_uint8, sigma,kernel, sf):
    img_H = img_clean_uint8
    img_L = img_clean_uint8
    k = kernel.squeeze(0)
    k = k.squeeze(0)
    x = img_H.squeeze(0)
    x = x.squeeze(0)

    out = np.zeros([img_H.shape[0],img_H.shape[1], int(img_H.shape[2] / sf), int(img_H.shape[3] / sf)])

    for i in range(3):
        a = torch.zeros([x.shape[1],x.shape[2],1])
        a[:,:,0] = x[i,:,:]
        b = sr_old.classical_degradation(a,k,sf)
        b = torch.from_numpy(b)
        out[0,i,:,:] = b[:,:,0]

    img_L = torch.from_numpy(out).float()
    np.random.seed(seed=0)

    # noise = np.random.normal(0, 1, img_L.shape)*sigma / 255
    # img_L += noise
    if sigma>0:
        img_L = np.float32(np.random.poisson(img_L * sigma) / sigma)
        img_L = torch.from_numpy(img_L)

    initial_uv = img_L
    return initial_uv, img_L, img_H






def search_args():
    device = 'cuda'
    

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

    search_range = {}

    sigma = 7.65
    sf = 4
    utils_logger.logger_info('sisr', log_path='log/sigma_{}/logger.log'.format(sigma))
    logger = logging.getLogger('sisr')
    logger.info('sigma = {}'.format(sigma))

    ## nb=101

    search_range[0.13] = [45] # ????

    search_level = [0.13]
    


    # kernel_fp = '/home/dlwei/Documents/pnp_jacobian/kernels/sisr_kernel.png'
    # kernel = util.imread_uint(kernel_fp,1)
    # kernel = util.single2tensor3(kernel).unsqueeze(0) / 255.
    # kernel = kernel / torch.sum(kernel)



    

    psnr_save_root  = 'log/' + 'sigma_' + str(sigma) + '/psnr'
    ssim_save_root  = 'log/' + 'sigma_' + str(sigma) + '/ssim'
    image_save_root = 'log/' + 'sigma_' + str(sigma) + '/image'
    if not os.path.exists(psnr_save_root):
        os.makedirs(psnr_save_root)    
    if not os.path.exists(ssim_save_root):
        os.makedirs(ssim_save_root)   
    if not os.path.exists(image_save_root):
        os.makedirs(image_save_root)


    for denoisor_level in search_level:
        logger.info('========================================')
        logger.info('denoisor_level: {}'.format(denoisor_level))
        logger.info('========================================')
        for sigma2 in [1.]: 
            # for lamb in range(*search_range[denoisor_level]):
            # len_lamb = len(search_range[denoisor_level])
            # print(len_lamb)
            # psnr_matrix = [0] * nb
            # ssim_matrix = [0] * nb
            for lamb in search_range[denoisor_level]:
                logger.info('==================')
                logger.info('lamb: {}'.format(lamb))

                dataset_psnr = None
                dataset_ssim = None
                image_paths = util.get_image_paths(dataset_root)
                image_number = len(image_paths)
                for ii in range(0,image_number):
                    fp = image_paths[ii]


                    kernel = create_gaussian_kernel(11, 1)
                    kernel = kernel.unsqueeze(0).unsqueeze(0)
                    
                    img_H = util.imread_uint(fp, 3)
                    img_H = util.single2tensor3(img_H).unsqueeze(0) /255.
                    initial_uv, img_L, img_H = gen_data(img_H, sigma,kernel, sf)
                    
                    initial_uv = initial_uv.to(device)
                    img_L = img_L.to(device)
                    img_H = img_H.to(device)
                    kernel = kernel.to(device)

                    with torch.no_grad():
                        img_L, img_H = img_L.to(device), img_H.to(device)
                        kernel = kernel.to(device)
                        model(kernel, initial_uv, img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5, sf)

                    cur_psnr = np.array(model.res['psnr'])
                    print(np.max(cur_psnr))
                    cur_ssim = np.array(model.res['ssim'])
                    if dataset_psnr is None:
                        dataset_psnr = cur_psnr
                        dataset_ssim = cur_ssim
                    else:
                        dataset_psnr += cur_psnr
                        dataset_ssim += cur_ssim

                dataset_psnr /= image_number
                dataset_ssim /= image_number
                print(dataset_psnr.shape)

                cur_avg_psnr = np.max(dataset_psnr)
                cur_avg_ssim = np.max(dataset_ssim)
                logger.info("PSNR: {:.4f}".format(cur_avg_psnr))
                logger.info("SSIM: {:.4f}".format(cur_avg_ssim))
                psnr_save_pth = psnr_save_root + '/level' + str(denoisor_level) + '_lamb' + str(lamb) + '_psnr' + str(cur_avg_psnr)[:7] + '.png'
                ssim_save_pth = ssim_save_root + '/level' + str(denoisor_level) + '_lamb' + str(lamb) + '_psnr' + str(cur_avg_psnr)[:7] + '.png'
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
    logger.info('max_psnr: {:.4f}'.format(max_psnr))
    logger.info('max_ssim: {:.4f}'.format(max_ssim))
    logger.info('level: {}'.format(max_level))
    logger.info('lamb: {}'.format(max_lamb))
    return max_psnr, max_level, max_lamb


# plot_psnr(denoisor_level, lamb, peak, sf)

#  0046.png, s=2, 
# step = 0.33, CoCo-ADMM
# plot_psnr(0.12, 2500, 300, 2) # 25.35
plot_psnr(0.1, 1500, 300, 2) # 25.36
# plot_psnr(0.05, 350, 300, 2) # 25.16




# max_psnr, max_level, max_lamb = search_args() 





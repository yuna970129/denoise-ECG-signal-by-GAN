import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader_ecg import BasicDataset
from torch.utils.data import DataLoader

class generator(nn.Module):
    def __init__(self, input_dim=500, output_dim=500, input_size=32, z_dim=100):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.z_dim = z_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024), # input_dim = 500
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 16)),
            nn.BatchNorm1d(128 * (self.input_size // 16)),
            nn.ReLU(),
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 4, 2, 1), #self.input_dim=500
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(1, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(64248, 500), 
            nn.BatchNorm1d(500),
            nn.ReLU(),
        )
        
        utils.initialize_weights(self)

    def forward(self, input, condition):
        x = condition.float()
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, input], 1)
        x = x.unsqueeze(1)
        x = self.deconv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc2(x)
        x = x.unsqueeze(1)

        return x

class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32, condition_dim=1):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim + self.condition_dim, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 125, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2),
            nn.Linear(500, self.output_dim), # 1
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, condition):
        x = torch.cat([input, condition], 1)
        x = x.float()
        x = self.conv(x)
        x = x.view(-1, 64 * 125)
        x = self.fc(x)
        return x

class DCLSGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.output_dim = 500
        self.root_dir = root_dir='./' 
        condition_dim = 1
        
        # load dataset
        data = BasicDataset(root_dir=self.root_dir)
        self.data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)

        # networks init
        self.G = generator(input_dim=500, output_dim=self.output_dim, input_size=self.input_size, z_dim=self.z_dim)
        self.D = discriminator(input_dim=1, output_dim=1, input_size=self.input_size, condition_dim=1)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()
                

    def train(self):
        # loss 저장할 list
        D_loss_list = []
        G_loss_list = []
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                
                x_ = x_.unsqueeze(1) # clear
                y_ = y_.unsqueeze(1) # noisy
                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_, y_ = x_.cuda(), z_.cuda(), y_.cuda()
                
                
                # update D network
                self.D_optimizer.zero_grad()

            # clean과 noisy 비교
                D_real = self.D(x_, y_)
                D_real_loss = self.MSE_loss(D_real, self.y_real_)

            # noisy로 denoised 만들기
                G_ = self.G(z_, y_)
            
            # denoised랑 noisy랑 비교
                D_fake = self.D(G_, y_)
                D_fake_loss = self.MSE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_)
                D_fake = self.D(G_, y_)
                G_loss = self.MSE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()
                
                # loss 계산
                D_loss_list.append(D_loss.item())
                G_loss_list.append(G_loss.item())

            print("Epoch: [%2d] D_loss: %.8f, G_loss: %.8f" %
                  ((epoch + 1), D_loss.item(), G_loss.item()))
            
            # signal 출력
            if (epoch+1) % 5 == 0 : # 5 epoch마다 signal 그래프 저장
                x_axis = np.linspace(0, 499, 500) # x축
                plt.plot(x_axis, x_.view(-1, 500)[0].detach().cpu().numpy(), label='clean') # clean
                plt.plot(x_axis, y_.view(-1, 500)[0].detach().cpu().numpy(), label='noisy') # noisy
                plt.plot(x_axis, G_.view(-1, 500)[0].detach().cpu().numpy(), label='generated') # generated
                plt.legend(loc='upper right')
                plt.savefig('%d_epoch_signal.png'%(epoch+1))
                plt.cla()
                plt.clf()
                plt.close()

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
        # loss 그래프 저장 
        total_bin = np.linspace(1, len(G_loss_list)-1, len(G_loss_list))
        plt.plot(total_bin, G_loss_list, label='Generator Loss') # G loss
        plt.plot(total_bin, D_loss_list, label='Discriminator Loss') # D loss
        plt.legend(loc='upper right')
        plt.savefig('%Final_Loss_Graph.png')
        plt.cla()
        plt.clf()
        plt.close()

        
        self.save()
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.randint(-1, 5, (self.batch_size, self.output_dim))
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
        
    

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
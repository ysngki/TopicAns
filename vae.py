import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


# VAE model
class VAE(nn.Module):
    def __init__(self, bow_dim, n_topic=20, dropout=0.0):

        super(VAE, self).__init__()

        self.id2token = None
        self.n_topic = n_topic

        encode_dims = [bow_dim, 1024, 512, n_topic]
        decode_dims = [n_topic, 512, bow_dim]

        self.encoder = nn.ModuleDict({
            f'enc_{i}':nn.Linear(encode_dims[i],encode_dims[i+1]) 
            for i in range(len(encode_dims)-2)
        })

        self.fc_mu = nn.Linear(encode_dims[-2],encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2],encode_dims[-1])

        self.decoder = nn.ModuleDict({
            f'dec_{i}':nn.Linear(decode_dims[i],decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })

        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1],encode_dims[-1])
        
    def encode(self, x):
        hid = x
        for _, layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, log_var

    def inference(self,x):
        mu, log_var = self.encode(x)
        theta = torch.softmax(x,dim=1)
        return theta
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        hid = z
        for i,(_,layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i<len(self.decoder)-1:
                hid = F.relu(self.dropout(hid))
        return hid
    
    def show_topic_words(self, device, topic_id=None, topK=30, dictionary=None):
        topic_words = []
        idxes = torch.eye(self.n_topic).to(device)

        word_dist = self.decode(idxes)
        word_dist = torch.softmax(word_dist,dim=1)
        vals, indices = torch.topk(word_dist,topK,dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()

        if self.id2token == None and dictionary != None:
            self.id2token = {v:k for k,v in dictionary.token2id.items()}
        if topic_id==None:
            for i in range(self.n_topic):
                topic_words.append([self.id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([self.id2token[idx] for idx in indices[topic_id]])

        return topic_words

    def forward(self, x, collate_fn=None):
        mu, log_var = self.encode(x)

        _theta = self.reparameterize(mu, log_var)
        _theta = self.fc1(_theta) 

        if collate_fn!=None:
            theta = collate_fn(_theta)
        else:
            theta = _theta

        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var

# if __name__ == '__main__':
#     model = VAE(encode_dims=[1024,512,256,20],decode_dims=[20,128,768,1024])
#     model = model.cuda()
#     inpt = torch.randn(234,1024).cuda()
#     out,mu,log_var = model(inpt)
#     print(out.shape)
#     print(mu.shape)

# WAE model
class WAE(nn.Module):
    def __init__(self, bow_dim, n_topic=20, dropout=0.0, nonlin='relu'):

        super(WAE, self).__init__()

        self.id2token = None
        self.n_topic = n_topic

        encode_dims = [bow_dim, 1024, 512, n_topic]
        decode_dims = [n_topic, 512, bow_dim]

        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i+1])
            for i in range(len(encode_dims)-1)
        })

        self.decoder = nn.ModuleDict({
            f'dec_{i}': nn.Linear(decode_dims[i], decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })

        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[nonlin]
        self.z_dim = encode_dims[-1]
        
    def encode(self, x):
        hid = x
        for i, (_,layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.encoder)-1:
                hid = self.nonlin(hid)
        return hid

    def decode(self, z):
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder)-1:
                hid = self.nonlin(self.dropout(hid))
        return hid

    def show_topic_words(self, device, topic_id=None, topK=30, dictionary=None):
        topic_words = []
        idxes = torch.eye(self.n_topic).to(device)

        word_dist = self.decode(idxes)
        word_dist = F.softmax(word_dist, dim=1)
        vals, indices = torch.topk(word_dist, topK, dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()

        if self.id2token==None and dictionary!=None:
            self.id2token = {v:k for k,v in dictionary.token2id.items()}
        if topic_id == None:
            for i in range(self.n_topic):
                topic_words.append([self.id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([self.id2token[idx] for idx in indices[topic_id]])

        return topic_words

    def forward(self, x, collate_fn=None):
        z = self.encode(x)
        theta = F.softmax(z, dim=1)
        x_reconst = self.decode(theta)
        return x_reconst, theta
    
    def sample(self, dist='dirichlet', batch_size=256, dirichlet_alpha=0.1, ori_data=None):
        if dist == 'dirichlet':
            z_true = np.random.dirichlet(
                np.ones(self.z_dim)*dirichlet_alpha, size=batch_size)
            z_true = torch.from_numpy(z_true).float()
            return z_true
        elif dist == 'gaussian':
            z_true = np.random.randn(batch_size, self.z_dim)
            z_true = torch.softmax(torch.from_numpy(z_true), dim=1).float()
            return z_true
        elif dist == 'gmm_std':
            odes = np.eye(self.z_dim)*20
            ides = np.random.randint(low=0, high=self.z_dim, size=batch_size)
            mus = odes[ides]
            sigmas = np.ones((batch_size, self.z_dim))*0.2*20
            z_true = np.random.normal(mus, sigmas)
            z_true = F.softmax(torch.from_numpy(z_true).float(), dim=1)
            return z_true
        elif dist=='gmm_ctm' and ori_data!=None:
            with torch.no_grad():
                hid_vecs = self.encode(ori_data).cpu().numpy()
                gmm = GaussianMixture(n_components=self.z_dim,covariance_type='full', max_iter=200)
                gmm.fit(hid_vecs)
                #hid_vecs = torch.from_numpy(hid_vecs).to(self.device)
                gmm_spls, _ = gmm.sample(n_samples=len(ori_data))
                theta_prior = torch.from_numpy(gmm_spls).float()
                theta_prior = F.softmax(theta_prior,dim=1)
                return theta_prior
        else:
            return self.sample(dist='dirichlet',batch_size=batch_size)

    def mmd_loss(self, x, y, device, t=0.1, kernel='diffusion'):
        '''
        computes the mmd loss with information diffusion kernel
        :param x: batch_size * latent dimension
        :param y:
        :param t:
        :return:
        '''
        eps = 1e-6
        n, d = x.shape
        if kernel == 'tv':
            sum_xx = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(i+1, n):
                    sum_xx = sum_xx + torch.norm(x[i]-x[j], p=1).to(device)
            sum_xx = sum_xx / (n * (n-1))

            sum_yy = torch.zeros(1).to(device)
            for i in range(y.shape[0]):
                for j in range(i+1, y.shape[0]):
                    sum_yy = sum_yy + torch.norm(y[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))

            sum_xy = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(y.shape[0]):
                    sum_xy = sum_xy + torch.norm(x[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (n * y.shape[0])
        else:
            qx = torch.sqrt(torch.clamp(x, eps, 1))
            qy = torch.sqrt(torch.clamp(y, eps, 1))
            xx = torch.matmul(qx, qx.t())
            yy = torch.matmul(qy, qy.t())
            xy = torch.matmul(qx, qy.t())

            def diffusion_kernel(a, tmpt, dim):
                # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
                return torch.exp(-torch.acos(a).pow(2)) / tmpt

            off_diag = 1 - torch.eye(n).to(device)
            k_xx = diffusion_kernel(torch.clamp(xx, 0, 1-eps), t, d-1)
            k_yy = diffusion_kernel(torch.clamp(yy, 0, 1-eps), t, d-1)
            k_xy = diffusion_kernel(torch.clamp(xy, 0, 1-eps), t, d-1)
            sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
            sum_xy = 2 * k_xy.sum() / (n * n)
        return sum_xx + sum_yy - sum_xy
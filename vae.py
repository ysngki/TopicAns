import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    def show_topic_words(self, device, topic_id=None, topK=15, dictionary=None):
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
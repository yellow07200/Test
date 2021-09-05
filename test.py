model = VGAE().to(device)
model.train()
optimizer = Adam(model.parameters(), lr=args.learning_rate)

for epoch in range(args.num_epoch):
    batch_c = 0
    running_acc = 0
    running_loss = 0
    running_log_lik = 0
    running_kl = 0
    for data in train_loader:
        batch_c = batch_c + 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        adj_norm= data.edge_index
        features = data.x

        optimizer.zero_grad()
        A_pred, z = model(features, adj_norm)

        # generator loss
        criterion = torch.nn.BCELoss(weight=weight_tensor)
        log_lik = norm*criterion(A_pred.cpu().view(-1), adj_label.to_dense().cpu().view(-1))
        loss = log_lik 
        running_log_lik += log_lik
        x_decoded_mean = A_pred.view(-1,)
        scaling_factor = x_decoded_mean.shape[0]

        # latent loss
        kl_divergence = -100 * 0.5 / A_pred.size(0) * torch.mean(torch.sum(1 + 2 * model.logstd - model.mean.pow(2) - (model.logstd).exp().pow(2), 1))
        running_kl += kl_divergence
        loss = log_lik + kl_divergence 

        running_loss += loss.item()
        loss.backward()
        optimizer.step()     

        train_acc = get_acc(A_pred.cpu(), adj_label.cpu())
        running_acc += train_acc

    train_acc = get_acc(A_pred.cpu(), adj_label.cpu())#adj_norm)#
    running_acc += train_acc

    running_acc = running_acc/ batch_c
    running_loss = running_loss/ batch_c
    running_log_lik = running_log_lik/ batch_c
    running_kl = running_kl/batch_c


class VGAE(nn.Module):
	def __init__(self):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim) #, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim,  activation=lambda x:x) #
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, activation=lambda x:x) #

	def encode(self, X, adj):
		hidden = self.base_gcn(X, adj)
		self.mean = self.gcn_mean(hidden, adj).cpu()
		self.logstd = self.gcn_logstddev(hidden, adj).cpu()
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim).cpu()
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X, adj):
		# print('X: ', X.size())
		Z = self.encode(X, adj)
		A_pred = dot_product_decode(Z)
		return A_pred, Z

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, activation = F.leaky_relu, **kwargs): #adj,
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim)
		self.activation = activation
		self.bn1 = torch.nn.BatchNorm1d(output_dim)

	def forward(self, inputs, adj):
		x = inputs
		x = torch.tensor(x, dtype = torch.float32, requires_grad=True).cuda()
		x = torch.mm(x,self.weight)
		x = torch.mm(adj, x)
		outputs = self.activation(x)
		outputs = F.dropout(outputs, args.dropout)
		output_bn = self.bn1(outputs)

		return output_bn 

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

def dot_product_decode(Z):
	Z = F.dropout(Z, args.dropout)
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy
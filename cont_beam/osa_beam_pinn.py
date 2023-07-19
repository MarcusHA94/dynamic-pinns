import torch
import torch.nn as nn
import numpy as np

def max_mag_data(data,axis=None):
    if torch.is_tensor(data):
        if axis==None:
            data_max = torch.max(torch.max(torch.abs(data)))
        else:
            data_max = torch.max(torch.abs(data),dim=axis)[0]
    else:
        data_max = np.max(np.abs(data),axis=axis)
    return data_max

def normalise(data,norm_type="var",norm_dir="all"):
    if norm_type=="var":
        if len(data.shape)>1 and norm_dir=="axis":
            mean = data.mean(axis=0)
            std = data.std(axis=0)
        else:
            mean = data.mean()
            std = data.std()
        data_norm = (data-mean)/std
        return data_norm, (mean, std)
    elif norm_type=="range":
        if len(data.shape)>1 and norm_dir=="axis":
            dmax = max_mag_data(data,axis=0)
        elif len(data.shape)>1 and norm_dir=="all":
            dmax = max_mag_data(data,None)
        else:
            dmax = max_mag_data(data)
        data_norm = data/dmax
        return data_norm, dmax

class osa_pinn_beam(nn.Module):

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wdd0, x, t):
        X = torch.cat((w0.view(-1,1), wdd0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y = self.net(X)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(0, self.l, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wdd = config["alphas"]['wdd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wdd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]-1
        n_x_obs = x_data.shape[0]
        n_x_col = self.ncx*(n_x_obs+1)

        # Observation set in matrix form (uses displacement one time point ahead)
        self.w0_obs = w_data[:,:-1].reshape(-1,1)
        self.wdd0_obs = wdd_data[:,:-1].reshape(-1,1)
        self.x_obs = x_data.repeat(1, n_t_obs)
        self.t_obs = self.T_hat * torch.ones((n_x_obs, n_t_obs)).reshape(-1,1)
        self.yy_obs = w_data[:,1:].reshape(-1,1)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        w0_col = torch.zeros((n_x_col, n_t_obs*self.nct))
        wdd0_col = torch.zeros((n_x_col, n_t_obs*self.nct))
        
        x_col = torch.zeros((n_x_col, n_t_obs*self.nct))
        t_col = torch.zeros((n_x_col, n_t_obs*self.nct))
        x_pred = torch.zeros((n_x_col, n_t_obs*self.nct))
        t_pred = torch.zeros((n_x_col, n_t_obs*self.nct))

        x_col_vec = torch.zeros((n_x_col,1))
        x_col_vec[:self.ncx,0] = self.x_wind.clone()
        for i in range(1,n_x_obs+1):
            x_col_vec[self.ncx*i:self.ncx*(i+1),0] = x_data[i-1,0] + self.x_wind.clone()

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data[i,j].item()*torch.ones((self.ncx,self.nct))
                wdd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wdd_data[i,j].item()*torch.ones((self.ncx,self.nct))
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.x_wind.clone().repeat(self.nct,1)
                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.clone().repeat(self.ncx,1).T

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data[i,j] + self.x_wind.clone().repeat(self.nct,1).T
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data[i,j] + self.t_wind.clone().repeat(self.ncx,1)

        self.w0_col = w0_col.reshape(-1,1).requires_grad_()
        self.wdd0_col = wdd0_col.reshape(-1,1).requires_grad_()
        self.x_col = x_col.reshape(-1,1).requires_grad_()
        self.t_col = t_col.reshape(-1,1).requires_grad_()

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        self.ic_ids = torch.argwhere(t_col[:,0]==torch.tensor(0.0))
        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self):

        # generate prediction at observation points
        wh_obs_hat = self.forward(self.w0_obs, self.wdd0_obs, self.x_obs, self.t_obs)
        R_obs = wh_obs_hat - self.yy_obs

        # generate prediction over prediction horizon
        wh_coll_hat = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col)

        # retrieve derivatives
        dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
        dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
        dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
        dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

        dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
        dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        # Initial condition residual
        R_ic1 = self.alpha_wdd * self.wdd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t**2)) * dt2[self.ic_ids]
        R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]

        # retrieve pde loss parameters
        match self.param_type:
            case "constant":
                self.m_hat = self.pde_alphas["dt2"]
                self.k_hat = self.pde_alphas["dx4"]
            case "variable":
                self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

        R_pde = self.m_hat * dt2 + self.k_hat * dx4

        # boundary condition residual
        R_bc = wh_coll_hat[self.bc_ids]

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde,
            "R_bc" : R_bc,
            "R_ic1" : R_ic1,
            "R_ic2" : R_ic2,
        }
    
    def loss_func(self, lambds):
        residuals = self.calc_residuals()
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic1 = residuals["R_ic1"]
        R_ic2 = residuals["R_ic2"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * (torch.mean(R_bc**2))
        L_ic = lambds['ic'] * (torch.mean(R_ic1**2) + torch.mean(R_ic2**2))
        loss = L_obs + L_pde + L_bc + L_ic

        return loss, [L_obs, L_pde, L_bc, L_ic]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col)
        return wp
 

class osa_pinn_beam_mode(nn.Module):
    """ this version does the mode estimator thing """

    def __init__(self, configuration):
        super().__init__()
        self.activation = nn.Tanh

        self.configure(**configuration)

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wd0, x0, x, t):
        X = torch.cat((w0.view(-1,1), wd0.view(-1,1), x0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y = self.net(X)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(0, self.L, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.true_ic_func = config["ic_func"]  # true initial condition as a function of x

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wd = config["alphas"]['wd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]
        n_x_obs = x_data.shape[0]

        # add zero condition data at boundaries
        w_data_ = torch.cat((torch.zeros(1,n_t_obs),w_data,torch.zeros(1,n_t_obs)),dim=0)
        wd_data_ = torch.cat((torch.zeros(1,n_t_obs),wd_data,torch.zeros(1,n_t_obs)),dim=0)
        x_data_ = torch.cat((torch.zeros(1,n_t_obs),x_data,self.l*torch.ones(1,n_t_obs)),dim=0)
        t_data_ = torch.cat((t_data[:-1,:],t_data[:3,:]),dim=0)

        w_data_1 = w_data_[:-1,1:]  # x=0,t=T
        w_data_2 = w_data_[1:,1:]   # x=L,t=T
        w_data_3 = w_data_[1:,:-1]  # x=L,t=0

        n_t_obs -= 1
        n_x_obs += 1

        # Observation set in matrix form (uses displacement one time point ahead)
        w0_obs_mat = torch.repeat_interleave(w_data_[:-1,:-1], 3, dim=0)
        wd0_obs_mat = torch.repeat_interleave(wd_data_[:-1,:-1], 3, dim=0)
        x0_obs_mat = torch.repeat_interleave(x_data_[:-1,:-1], 3, dim=0)
        x_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            x_obs_mat[3*i+1,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
            x_obs_mat[3*i+2,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
        t_obs_mat = torch.cat((
            self.T_hat * torch.ones((2, n_t_obs)),
            torch.zeros((1,n_t_obs))
        ),dim=0).repeat((n_x_obs,1))
        yy_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            yy_obs_mat[3*i,:] = w_data_1[i,:]
            yy_obs_mat[3*i+1,:] = w_data_2[i,:]
            yy_obs_mat[3*i+2,:] = w_data_3[i,:]

        self.w0_obs = w0_obs_mat.reshape(-1).to(self.device)
        self.wd0_obs = wd0_obs_mat.reshape(-1).to(self.device)
        self.x0_obs = x0_obs_mat.reshape(-1).to(self.device)
        self.x_obs = x_obs_mat.reshape(-1).to(self.device)
        self.t_obs = t_obs_mat.reshape(-1).to(self.device)
        self.yy_obs = yy_obs_mat.reshape(-1).to(self.device)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        self.n_col_x = n_x_obs*self.ncx
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x, self.n_col_t))
        wd0_col = torch.zeros((self.n_col_x, self.n_col_t))
        x0_col = torch.zeros((self.n_col_x, self.n_col_t))
        
        x_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in x
        t_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in time
        x_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true x coordinate of cols)
        t_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true time of cols)

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                wd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wd_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x_wind = torch.linspace(0,x_data_[i+1,0]-x_data_[i,0],self.ncx)
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_wind.view(self.ncx,1).repeat(1,self.nct)

                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.view(1,self.nct).repeat(self.ncx,1)

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j] + x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data_[i,j] + self.t_wind.view(1,self.nct).repeat(self.ncx,1)

        self.w0_col = w0_col.reshape(-1).requires_grad_().to(self.device)
        self.wd0_col = wd0_col.reshape(-1).requires_grad_().to(self.device)
        self.x0_col = x0_col.reshape(-1).requires_grad_().to(self.device)

        self.x_col = x_col.reshape(-1).requires_grad_().to(self.device)
        self.t_col = t_col.reshape(-1).requires_grad_().to(self.device)

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        pred_cat = torch.cat((self.x_pred, self.t_pred), dim=1)
        unqs, counts = torch.unique(pred_cat, return_counts=True, dim=0)
        self.cc_pairs = torch.zeros((sum(counts>1),2), dtype=torch.int32)
        self.cc_pairs = []
        repeated_groups = unqs[counts>1]
        for i, repeated_group in enumerate(repeated_groups):
            repeated_idx = torch.argwhere(torch.all(pred_cat==repeated_group, dim=1)).reshape(-1)
            if (repeated_idx.shape[0]>2):
                repeated_idx_ = torch.combinations(repeated_idx,r=2)
                [self.cc_pairs.append(repeated_idx_[j,:].view(-1,1)) for j in range(repeated_idx_.shape[0])]
            else:
                self.cc_pairs.append(repeated_idx.view(-1,1))
        self.cc_pairs = torch.cat(self.cc_pairs, dim=1).T

        self.ic_ids = torch.argwhere((self.t_col==torch.tensor(0.0)) * (self.x_col==torch.tensor(0.0)))

        self.true_ic_ids = torch.argwhere((self.t_pred==torch.tensor(0.0)))[:,0]
        self.true_ic = self.true_ic_func(self.x_pred[self.true_ic_ids]*self.alpha_x)/self.alpha_w

        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self, switches):

        if switches['obs']:
            # generate prediction at observation points
            wh_obs_hat = torch.zeros(self.w0_obs.shape[0]).to(self.device)
            wh_obs_hat = self.forward(self.w0_obs, self.wd0_obs, self.x0_obs, self.x_obs, self.t_obs).squeeze()
            R_obs = wh_obs_hat - self.yy_obs
        else:
            R_obs = torch.zeros(10).to(self.device)

        # generate prediction over prediction horizon
        wh_coll_hat = torch.zeros(self.w0_col.shape[0]).to(self.device)
        wh_coll_hat[:] = self.forward(self.w0_col, self.wd0_col, self.x0_col, self.x_col, self.t_col).squeeze()

        if switches['pde'] or switches['ic']:

            # retrieve derivatives
            dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
            dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        if switches['pde'] or switches['bc']:
            dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
            dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
        
        if switches['pde']:
            dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

            # retrieve pde loss parameters
            match self.param_type:
                case "constant":
                    self.m_hat = self.pde_alphas["dt2"]
                    self.k_hat = self.pde_alphas["dx4"]
                case "variable":
                    self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                    self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

            R_cc = torch.zeros((self.cc_pairs.shape[0],4))
            R_cc[:,0] = wh_coll_hat[self.cc_pairs[:,0]] - wh_coll_hat[self.cc_pairs[:,1]]
            R_cc[:,1] = (dx[self.cc_pairs[:,0]] - dx[self.cc_pairs[:,1]])
            R_cc[:,2] = (dx2[self.cc_pairs[:,0]] - dx2[self.cc_pairs[:,1]])
            # R_cc[:,3] = (dx3[self.cc_pairs[:,0]] - dx3[self.cc_pairs[:,1]])
            # R_cc[:,2] = (dx4[self.cc_pairs[:,0]] - dx4[self.cc_pairs[:,1]])
            R_cc[:,3] = (dt[self.cc_pairs[:,0]] - dt[self.cc_pairs[:,1]])
            # R_cc[:,6] = (dt2[self.cc_pairs[:,0]] - dt2[self.cc_pairs[:,1]])

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(10)
            R_cc = torch.zeros((10,2))

        if switches['ic']:
            # Initial condition residual
            R_ic1 = self.alpha_wd * self.wd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t)) * dt[self.ic_ids]
            R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]
            R_ic = torch.cat((R_ic1.view(-1,1), R_ic2.view(-1,1)),dim=1)
        else:
            R_ic = torch.zeros((10,2))

        if switches['bc']:
            # boundary condition residual
            R_bc1 = wh_coll_hat[self.bc_ids]
            R_bc2 = dx2[self.bc_ids]
            R_bc = torch.cat((R_bc1.view(-1,1), R_bc2.view(-1,1)), dim=1)
        else:
            R_bc = torch.zeros((10,2))

        if switches['tic']:
            R_tic = wh_coll_hat[self.true_ic_ids] - self.true_ic.view(-1)
        else:
            R_tic = torch.zeros(10)

        return {
            "R_obs" : R_obs.squeeze(),
            "R_pde" : R_pde.squeeze(),
            "R_bc" : R_bc.squeeze(),
            "R_ic" : R_ic.squeeze(),
            "R_tic" : R_tic.squeeze(),
            "R_cc" : R_cc.squeeze()
        }
    
    def set_loss_switches(self, lambds):

        switches = {}
        for key, value in lambds.items():
            if isinstance(value, list):
                if any([val>0.0 for val in value]):
                    switches[key] = True
                else:
                    switches[key] = False
            else:
                switches[key] = value>0.0
        self.switches = switches
    
    def loss_func(self, lambds):

        residuals = self.calc_residuals(self.switches)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic = residuals["R_ic"]
        R_tic = residuals["R_tic"]
        R_cc = residuals["R_cc"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * torch.sum(torch.mean(R_bc**2,dim=0),dim=0)
        L_ic = lambds['ic'] * torch.sum(torch.mean(R_ic**2,dim=0),dim=0)
        L_tic = lambds['tic'] * (torch.mean(R_tic**2))
        L_cc = lambds['cc'][0] * torch.mean(R_cc[:,0]**2) + lambds['cc'][1]*torch.mean(R_cc[:,1]**2)
        loss = L_obs + L_pde + L_tic + L_ic + L_bc + L_cc

        return loss, [L_obs, L_pde, L_tic, L_ic, L_bc, L_cc]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wd0_col, self.x0_col, self.x_col, self.t_col)
        return wp
    

class osa_pinn_beam_vel_2_1(nn.Module):
    """ version 2.1 includes the d2w/dx2 ics """

    def __init__(self, configuration):
        super().__init__()
        self.activation = nn.Tanh

        self.configure(**configuration)

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wd0, x0, x, t):
        X = torch.cat((w0.view(-1,1), wd0.view(-1,1), x0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y = self.net(X)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(0, self.L, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.true_ic_func = config["ic_func"]  # true initial condition as a function of x

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wd = config["alphas"]['wd']
        self.alpha_xdd = config["alphas"]['xdd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wd_data, xdd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]
        n_x_obs = x_data.shape[0]

        # add zero condition data at boundaries
        w_data_ = torch.cat((torch.zeros(1,n_t_obs),w_data,torch.zeros(1,n_t_obs)),dim=0)
        wd_data_ = torch.cat((torch.zeros(1,n_t_obs),wd_data,torch.zeros(1,n_t_obs)),dim=0)
        x_data_ = torch.cat((torch.zeros(1,n_t_obs),x_data,self.l*torch.ones(1,n_t_obs)),dim=0)
        xdd_data_ = torch.cat((torch.zeros(1,n_t_obs),xdd_data,torch.zeros(1,n_t_obs)),dim=0)
        t_data_ = torch.cat((t_data[:-1,:],t_data[:3,:]),dim=0)

        w_data_1 = w_data_[:-1,1:]  # x=0,t=T
        w_data_2 = w_data_[1:,1:]   # x=L,t=T
        w_data_3 = w_data_[1:,:-1]  # x=L,t=0

        n_t_obs -= 1
        n_x_obs += 1

        # Observation set in matrix form (uses displacement one time point ahead)
        w0_obs_mat = torch.repeat_interleave(w_data_[:-1,:-1], 3, dim=0)
        wd0_obs_mat = torch.repeat_interleave(wd_data_[:-1,:-1], 3, dim=0)
        x0_obs_mat = torch.repeat_interleave(x_data_[:-1,:-1], 3, dim=0)
        x_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            x_obs_mat[3*i+1,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
            x_obs_mat[3*i+2,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
        t_obs_mat = torch.cat((
            self.T_hat * torch.ones((2, n_t_obs)),
            torch.zeros((1,n_t_obs))
        ),dim=0).repeat((n_x_obs,1))
        yy_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            yy_obs_mat[3*i,:] = w_data_1[i,:]
            yy_obs_mat[3*i+1,:] = w_data_2[i,:]
            yy_obs_mat[3*i+2,:] = w_data_3[i,:]

        self.w0_obs = w0_obs_mat.reshape(-1).to(self.device)
        self.wd0_obs = wd0_obs_mat.reshape(-1).to(self.device)
        self.x0_obs = x0_obs_mat.reshape(-1).to(self.device)
        self.x_obs = x_obs_mat.reshape(-1).to(self.device)
        self.t_obs = t_obs_mat.reshape(-1).to(self.device)
        self.yy_obs = yy_obs_mat.reshape(-1).to(self.device)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        self.n_col_x = n_x_obs*self.ncx
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x, self.n_col_t))
        wd0_col = torch.zeros((self.n_col_x, self.n_col_t))
        x0_col = torch.zeros((self.n_col_x, self.n_col_t))
        xdd_col = torch.zeros((self.n_col_x, self.n_col_t))
        
        x_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in x
        t_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in time
        x_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true x coordinate of cols)
        t_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true time of cols)

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                wd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wd_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                xdd_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = xdd_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x_wind = torch.linspace(0,x_data_[i+1,0]-x_data_[i,0],self.ncx)
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_wind.view(self.ncx,1).repeat(1,self.nct)

                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.view(1,self.nct).repeat(self.ncx,1)

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j] + x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data_[i,j] + self.t_wind.view(1,self.nct).repeat(self.ncx,1)

        self.w0_col = w0_col.reshape(-1).requires_grad_().to(self.device)
        self.wd0_col = wd0_col.reshape(-1).requires_grad_().to(self.device)
        self.x0_col = x0_col.reshape(-1).requires_grad_().to(self.device)
        self.xdd_col = xdd_col.reshape(-1).requires_grad_().to(self.device)

        self.x_col = x_col.reshape(-1).requires_grad_().to(self.device)
        self.t_col = t_col.reshape(-1).requires_grad_().to(self.device)

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        pred_cat = torch.cat((self.x_pred, self.t_pred), dim=1)
        unqs, counts = torch.unique(pred_cat, return_counts=True, dim=0)
        self.cc_pairs = torch.zeros((sum(counts>1),2), dtype=torch.int32)
        self.cc_pairs = []
        repeated_groups = unqs[counts>1]
        for i, repeated_group in enumerate(repeated_groups):
            repeated_idx = torch.argwhere(torch.all(pred_cat==repeated_group, dim=1)).reshape(-1)
            if (repeated_idx.shape[0]>2):
                repeated_idx_ = torch.combinations(repeated_idx,r=2)
                [self.cc_pairs.append(repeated_idx_[j,:].view(-1)) for j in range(repeated_idx_.shape[0])]
            else:
                self.cc_pairs.append(repeated_idx)

        self.ic_ids = torch.argwhere((self.t_col==torch.tensor(0.0)) * (self.x_col==torch.tensor(0.0)))

        self.true_ic_ids = torch.argwhere((self.t_pred==torch.tensor(0.0)))[:,0]
        self.true_ic = self.true_ic_func(self.x_pred[self.true_ic_ids]*self.alpha_x)/self.alpha_w

        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self, switches):

        if switches['obs']:
            # generate prediction at observation points
            wh_obs_hat = torch.zeros(self.w0_obs.shape[0]).to(self.device)
            wh_obs_hat = self.forward(self.w0_obs, self.wd0_obs, self.x0_obs, self.x_obs, self.t_obs).squeeze()
            R_obs = wh_obs_hat - self.yy_obs
        else:
            R_obs = torch.zeros(10).to(self.device)

        # generate prediction over prediction horizon
        wh_coll_hat = torch.zeros(self.w0_col.shape[0]).to(self.device)
        wh_coll_hat[:] = self.forward(self.w0_col, self.wd0_col, self.x0_col, self.x_col, self.t_col).squeeze()

        if switches['pde'] or switches['ic']:

            # retrieve derivatives
            dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
            dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        if switches['pde'] or switches['bc']:
            dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
            dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
        
        if switches['pde']:
            dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

            # retrieve pde loss parameters
            match self.param_type:
                case "constant":
                    self.m_hat = self.pde_alphas["dt2"]
                    self.k_hat = self.pde_alphas["dx4"]
                case "variable":
                    self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                    self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

            R_cc = torch.zeros((len(self.cc_pairs),2))
            for i in range(len(self.cc_pairs)):
                R_cc[i,0] = (dx4[self.cc_pairs[i][0]] - dx4[self.cc_pairs[i][1]])
                R_cc[i,1] = (dt2[self.cc_pairs[i][0]] - dt2[self.cc_pairs[i][1]])

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(10)
            R_cc = torch.zeros((10,2))

        if switches['ic']:
            # Initial condition residual
            R_ic1 = self.alpha_wd * self.wd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t)) * dt[self.ic_ids]
            R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]
            R_ic3 = self.alpha_xdd * self.xdd_col[self.ic_ids] - (self.alpha_w/(self.alpha_x**2)) * dx2[self.ic_ids]
            R_ic = torch.cat((R_ic1.view(-1,1), R_ic2.view(-1,1), R_ic3.view(-1,1)),dim=1)
        else:
            R_ic = torch.zeros((10,3))

        if switches['bc']:
            # boundary condition residual
            R_bc1 = wh_coll_hat[self.bc_ids]
            R_bc2 = dx2[self.bc_ids]
            R_bc = torch.cat((R_bc1.view(-1,1), R_bc2.view(-1,1)), dim=1)
        else:
            R_bc = torch.zeros((10,2))

        if switches['tic']:
            R_tic = wh_coll_hat[self.true_ic_ids] - self.true_ic.view(-1)
        else:
            R_tic = torch.zeros(10)

        return {
            "R_obs" : R_obs.squeeze(),
            "R_pde" : R_pde.squeeze(),
            "R_bc" : R_bc.squeeze(),
            "R_ic" : R_ic.squeeze(),
            "R_tic" : R_tic.squeeze(),
            "R_cc" : R_cc.squeeze()
        }
    
    def set_loss_switches(self, lambds):

        switches = {}
        for key, value in lambds.items():
            if isinstance(value, list):
                if any([val>0.0 for val in value]):
                    switches[key] = True
                else:
                    switches[key] = False
            else:
                switches[key] = value>0.0
        self.switches = switches
    
    def loss_func(self, lambds):

        residuals = self.calc_residuals(self.switches)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic = residuals["R_ic"]
        R_tic = residuals["R_tic"]
        R_cc = residuals["R_cc"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * torch.sum(torch.mean(R_bc**2,dim=0),dim=0)
        L_ic = lambds['ic'] * torch.sum(torch.mean(R_ic**2,dim=0),dim=0)
        L_tic = lambds['tic'] * (torch.mean(R_tic**2))
        L_cc = lambds['cc'][0] * torch.mean(R_cc[:,0]**2) + lambds['cc'][1]*torch.mean(R_cc[:,1]**2)
        loss = L_obs + L_pde + L_tic + L_ic + L_bc + L_cc

        return loss, [L_obs, L_pde, L_tic, L_ic, L_bc, L_cc]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wd0_col, self.x0_col, self.x_col, self.t_col)
        return wp


class osa_pinn_beam_vel_2(nn.Module):
    """ version two includes the true initial condition and x0 """

    def __init__(self, configuration):
        super().__init__()
        self.activation = nn.Tanh

        self.configure(**configuration)

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wd0, x0, x, t, G=0.0, D=1.0):
        X = torch.cat((w0.view(-1,1), wd0.view(-1,1), x0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y_ = self.net(X)
        y = G + D * y_.view(-1)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(0, self.L, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.true_ic_func = config["ic_func"]  # true initial condition as a function of x

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wd = config["alphas"]['wd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]
        n_x_obs = x_data.shape[0]

        # add zero condition data at boundaries
        w_data_ = torch.cat((torch.zeros(1,n_t_obs),w_data,torch.zeros(1,n_t_obs)),dim=0)
        wd_data_ = torch.cat((torch.zeros(1,n_t_obs),wd_data,torch.zeros(1,n_t_obs)),dim=0)
        x_data_ = torch.cat((torch.zeros(1,n_t_obs),x_data,self.l*torch.ones(1,n_t_obs)),dim=0)
        t_data_ = torch.cat((t_data[:-1,:],t_data[:3,:]),dim=0)

        w_data_1 = w_data_[:-1,1:]  # x=0,t=T
        w_data_2 = w_data_[1:,1:]   # x=L,t=T
        w_data_3 = w_data_[1:,:-1]  # x=L,t=0

        n_t_obs -= 1
        n_x_obs += 1

        # Observation set in matrix form (uses displacement one time point ahead)
        w0_obs_mat = torch.repeat_interleave(w_data_[:-1,:-1], 3, dim=0)
        wd0_obs_mat = torch.repeat_interleave(wd_data_[:-1,:-1], 3, dim=0)
        x0_obs_mat = torch.repeat_interleave(x_data_[:-1,:-1], 3, dim=0)
        x_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            x_obs_mat[3*i+1,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
            x_obs_mat[3*i+2,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
        t_obs_mat = torch.cat((
            self.T_hat * torch.ones((2, n_t_obs)),
            torch.zeros((1,n_t_obs))
        ),dim=0).repeat((n_x_obs,1))
        yy_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            yy_obs_mat[3*i,:] = w_data_1[i,:]
            yy_obs_mat[3*i+1,:] = w_data_2[i,:]
            yy_obs_mat[3*i+2,:] = w_data_3[i,:]

        self.w0_obs = w0_obs_mat.reshape(-1).to(self.device)
        self.wd0_obs = wd0_obs_mat.reshape(-1).to(self.device)
        self.x0_obs = x0_obs_mat.reshape(-1).to(self.device)
        self.x_obs = x_obs_mat.reshape(-1).to(self.device)
        self.t_obs = t_obs_mat.reshape(-1).to(self.device)
        self.yy_obs = yy_obs_mat.reshape(-1).to(self.device)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        self.n_col_x = n_x_obs*self.ncx
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x, self.n_col_t))
        wd0_col = torch.zeros((self.n_col_x, self.n_col_t))
        x0_col = torch.zeros((self.n_col_x, self.n_col_t))
        
        x_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in x
        t_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in time
        x_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true x coordinate of cols)
        t_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true time of cols)

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                wd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wd_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x_wind = torch.linspace(0,x_data_[i+1,0]-x_data_[i,0],self.ncx)
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_wind.view(self.ncx,1).repeat(1,self.nct)

                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.view(1,self.nct).repeat(self.ncx,1)

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j] + x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data_[i,j] + self.t_wind.view(1,self.nct).repeat(self.ncx,1)

        self.w0_col = w0_col.reshape(-1).requires_grad_().to(self.device)
        self.wd0_col = wd0_col.reshape(-1).requires_grad_().to(self.device)
        self.x0_col = x0_col.reshape(-1).requires_grad_().to(self.device)

        self.x_col = x_col.reshape(-1).requires_grad_().to(self.device)
        self.t_col = t_col.reshape(-1).requires_grad_().to(self.device)

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        pred_cat = torch.cat((self.x_pred, self.t_pred), dim=1)
        unqs, counts = torch.unique(pred_cat, return_counts=True, dim=0)
        self.cc_pairs = torch.zeros((sum(counts>1),2), dtype=torch.int32)
        self.cc_pairs = []
        repeated_groups = unqs[counts>1]
        for i, repeated_group in enumerate(repeated_groups):
            repeated_idx = torch.argwhere(torch.all(pred_cat==repeated_group, dim=1)).reshape(-1)
            if (repeated_idx.shape[0]>2):
                repeated_idx_ = torch.combinations(repeated_idx,r=2)
                [self.cc_pairs.append(repeated_idx_[j,:].view(-1,1)) for j in range(repeated_idx_.shape[0])]
            else:
                self.cc_pairs.append(repeated_idx.view(-1,1))
        self.cc_pairs = torch.cat(self.cc_pairs, dim=1).T

        self.ic_ids = torch.argwhere((self.t_col==torch.tensor(0.0)) * (self.x_col==torch.tensor(0.0)))
        self.ec_ids = torch.argwhere((self.t_col==torch.tensor(0.0)) * (self.x_col==torch.tensor(0.0)))
        # self.G = torch.zeros_like(self.x_col)
        # self.G[self.ic_ids] = self.w0_col[self.ic_ids]
        # self.D = torch.ones_like(self.x_col)
        # self.D[self.ic_ids] = 0.0
        # self.G = self.G.requires_grad_()
        # self.D = self.D.requires_grad_()

        self.true_ic_ids = torch.argwhere((self.t_pred==torch.tensor(0.0)))[:,0]
        self.true_ic = self.true_ic_func(self.x_pred[self.true_ic_ids]*self.alpha_x)/self.alpha_w

        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self, switches):

        if switches['obs']:
            # generate prediction at observation points
            wh_obs_hat = torch.zeros(self.w0_obs.shape[0]).to(self.device)
            wh_obs_hat = self.forward(self.w0_obs, self.wd0_obs, self.x0_obs, self.x_obs, self.t_obs).squeeze()
            R_obs = wh_obs_hat - self.yy_obs
        else:
            R_obs = torch.zeros(10).to(self.device)

        # generate prediction over prediction horizon
        wh_coll_hat = torch.zeros(self.w0_col.shape[0]).to(self.device)
        wh_coll_hat = self.forward(self.w0_col, self.wd0_col, self.x0_col, self.x_col, self.t_col).squeeze()

        if switches['pde'] or switches['ic']:

            # retrieve derivatives
            dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
            dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        if switches['pde'] or switches['bc']:
            dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
            dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
        
        if switches['pde']:
            dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

            # retrieve pde loss parameters
            match self.param_type:
                case "constant":
                    self.m_hat = self.pde_alphas["dt2"]
                    self.k_hat = self.pde_alphas["dx4"]
                case "variable":
                    self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                    self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

            R_cc = torch.zeros((self.cc_pairs.shape[0],4))
            R_cc[:,0] = wh_coll_hat[self.cc_pairs[:,0]] - wh_coll_hat[self.cc_pairs[:,1]]
            R_cc[:,1] = (dx[self.cc_pairs[:,0]] - dx[self.cc_pairs[:,1]])
            R_cc[:,2] = (dx2[self.cc_pairs[:,0]] - dx2[self.cc_pairs[:,1]])
            # R_cc[:,3] = (dx3[self.cc_pairs[:,0]] - dx3[self.cc_pairs[:,1]])
            # R_cc[:,2] = (dx4[self.cc_pairs[:,0]] - dx4[self.cc_pairs[:,1]])
            R_cc[:,3] = (dt[self.cc_pairs[:,0]] - dt[self.cc_pairs[:,1]])
            # R_cc[:,6] = (dt2[self.cc_pairs[:,0]] - dt2[self.cc_pairs[:,1]])

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(10)
            R_cc = torch.zeros((10,2))

        if switches['ic']:
            # Initial condition residual
            R_ic1 = self.alpha_wd * self.wd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t)) * dt[self.ic_ids]
            R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]
            R_ic = torch.cat((R_ic1.view(-1,1), R_ic2.view(-1,1)),dim=1)
        else:
            R_ic = torch.zeros((10,2))

        if switches['bc']:
            # boundary condition residual
            R_bc1 = wh_coll_hat[self.bc_ids]
            R_bc2 = dx2[self.bc_ids]
            R_bc = torch.cat((R_bc1.view(-1,1), R_bc2.view(-1,1)), dim=1)
        else:
            R_bc = torch.zeros((10,2))

        if switches['tic']:
            R_tic = wh_coll_hat[self.true_ic_ids] - self.true_ic.view(-1)
        else:
            R_tic = torch.zeros(10)

        return {
            "R_obs" : R_obs.squeeze(),
            "R_pde" : R_pde.squeeze(),
            "R_bc" : R_bc.squeeze(),
            "R_ic" : R_ic.squeeze(),
            "R_tic" : R_tic.squeeze(),
            "R_cc" : R_cc.squeeze()
        }
    
    def set_loss_switches(self, lambds):

        switches = {}
        for key, value in lambds.items():
            if isinstance(value, list):
                if any([val>0.0 for val in value]):
                    switches[key] = True
                else:
                    switches[key] = False
            else:
                switches[key] = value>0.0
        self.switches = switches
    
    def loss_func(self, lambds):

        residuals = self.calc_residuals(self.switches)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic = residuals["R_ic"]
        R_tic = residuals["R_tic"]
        R_cc = residuals["R_cc"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * torch.sum(torch.mean(R_bc**2,dim=0),dim=0)
        L_ic = lambds['ic'] * torch.sum(torch.mean(R_ic**2,dim=0),dim=0)
        L_tic = lambds['tic'] * (torch.mean(R_tic**2))
        L_cc = lambds['cc'] * torch.sum(torch.mean(R_cc**2,dim=0),dim=0)
        loss = L_obs + L_pde + L_tic + L_ic + L_bc + L_cc

        return loss, [L_obs, L_pde, L_tic, L_ic, L_bc, L_cc]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wd0_col, self.x0_col, self.x_col, self.t_col)
        return wp


class osa_pinn_beam_vel(nn.Module):

    def __init__(self, configuration):
        super().__init__()
        self.activation = nn.Tanh

        self.configure(**configuration)

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wd0, x, t):
        X = torch.cat((w0.view(-1,1), wd0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y = self.net(X)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(0, self.L, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wd = config["alphas"]['wd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]
        n_x_obs = x_data.shape[0]

        # add zero condition data at boundaries
        w_data_ = torch.cat((torch.zeros(1,n_t_obs),w_data,torch.zeros(1,n_t_obs)),dim=0)
        # w_data_ = torch.cat((w_data,torch.zeros(1,n_t_obs)),dim=0)
        wd_data_ = torch.cat((torch.zeros(1,n_t_obs),wd_data,torch.zeros(1,n_t_obs)),dim=0)
        # wd_data_ = torch.cat((wd_data,torch.zeros(1,n_t_obs)),dim=0)
        x_data_ = torch.cat((torch.zeros(1,n_t_obs),x_data,self.l*torch.ones(1,n_t_obs)),dim=0)
        # x_data_ = torch.cat((x_data,self.L*torch.ones(1,n_t_obs)),dim=0)
        t_data_ = torch.cat((t_data[:-1,:],t_data[:3,:]),dim=0)

        w_data_1 = w_data_[:-1,1:]  # x=0,t=T
        w_data_2 = w_data_[1:,1:]   # x=L,t=T
        w_data_3 = w_data_[1:,:-1]  # x=L,t=0

        n_t_obs -= 1
        n_x_obs += 1
        # n_x_obs -= 1

        # Observation set in matrix form (uses displacement one time point ahead)
        w0_obs_mat = torch.repeat_interleave(w_data_[:-1,:-1], 3, dim=0)
        wd0_obs_mat = torch.repeat_interleave(wd_data_[:-1,:-1], 3, dim=0)
        x_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            x_obs_mat[3*i+1,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
            x_obs_mat[3*i+2,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
        t_obs_mat = torch.cat((
            self.T_hat * torch.ones((2, n_t_obs)),
            torch.zeros((1,n_t_obs))
        ),dim=0).repeat((n_x_obs,1))
        yy_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            yy_obs_mat[3*i,:] = w_data_1[i,:]
            yy_obs_mat[3*i+1,:] = w_data_2[i,:]
            yy_obs_mat[3*i+2,:] = w_data_3[i,:]

        self.w0_obs = w0_obs_mat.reshape(-1).to(self.device)
        self.wd0_obs = wd0_obs_mat.reshape(-1).to(self.device)
        self.x_obs = x_obs_mat.reshape(-1).to(self.device)
        self.t_obs = t_obs_mat.reshape(-1).to(self.device)
        self.yy_obs = yy_obs_mat.reshape(-1).to(self.device)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        self.n_col_x = n_x_obs*self.ncx
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x, self.n_col_t))
        wd0_col = torch.zeros((self.n_col_x, self.n_col_t))
        
        x_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in x
        t_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in time
        x_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true x coordinate of cols)
        t_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true time of cols)

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                wd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wd_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x_wind = torch.linspace(0,x_data_[i+1,0]-x_data_[i,0],self.ncx)
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_wind.view(self.ncx,1).repeat(1,self.nct)

                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.view(1,self.nct).repeat(self.ncx,1)

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j] + x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data_[i,j] + self.t_wind.view(1,self.nct).repeat(self.ncx,1)

        self.w0_col = w0_col.reshape(-1).requires_grad_().to(self.device)
        self.wd0_col = wd0_col.reshape(-1).requires_grad_().to(self.device)
        
        self.x_col = x_col.reshape(-1).requires_grad_().to(self.device)
        self.t_col = t_col.reshape(-1).requires_grad_().to(self.device)

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        pred_cat = torch.cat((self.x_pred, self.t_pred), dim=1)
        unqs, counts = torch.unique(pred_cat, return_counts=True, dim=0)
        self.cc_pairs = torch.zeros((sum(counts>1),2), dtype=torch.int32)
        self.cc_pairs = []
        repeated_groups = unqs[counts>1]
        for i, repeated_group in enumerate(repeated_groups):
            repeated_idx = torch.argwhere(torch.all(pred_cat==repeated_group, dim=1)).reshape(-1)
            if (repeated_idx.shape[0]>2):
                repeated_idx_ = torch.combinations(repeated_idx,r=2)
                [self.cc_pairs.append(repeated_idx_[j,:].view(-1)) for j in range(repeated_idx_.shape[0])]
            else:
                self.cc_pairs.append(repeated_idx)

        self.ic_ids = torch.argwhere((self.t_col==torch.tensor(0.0)) * (self.x_col==torch.tensor(0.0)))
        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self, switches):

        if switches['obs']:
            # generate prediction at observation points
            wh_obs_hat = torch.zeros(self.w0_obs.shape[0]).to(self.device)
            wh_obs_hat = self.forward(self.w0_obs, self.wd0_obs, self.x_obs, self.t_obs).squeeze()
            R_obs = wh_obs_hat - self.yy_obs
        else:
            R_obs = torch.zeros(10).to(self.device)

        # generate prediction over prediction horizon
        wh_coll_hat = torch.zeros(self.w0_col.shape[0]).to(self.device)
        wh_coll_hat[:] = self.forward(self.w0_col, self.wd0_col, self.x_col, self.t_col).squeeze()

        if switches['pde'] or switches['ic']:

            # retrieve derivatives
            dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
            dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        if switches['pde']:
            dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
            dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
            dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

            # retrieve pde loss parameters
            match self.param_type:
                case "constant":
                    self.m_hat = self.pde_alphas["dt2"]
                    self.k_hat = self.pde_alphas["dx4"]
                case "variable":
                    self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                    self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

            R_cc = torch.zeros((len(self.cc_pairs),2))
            for i in range(len(self.cc_pairs)):
                R_cc[i,0] = (dx4[self.cc_pairs[i][0]] - dx4[self.cc_pairs[i][1]])
                R_cc[i,1] = (dt2[self.cc_pairs[i][0]] - dt2[self.cc_pairs[i][1]])*self.m_hat

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(10)
            R_cc = torch.zeros((10,2))

        if switches['ic']:
            # Initial condition residual
            R_ic1 = self.alpha_wd * self.wd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t)) * dt[self.ic_ids]
            R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]
            R_ic = torch.cat((R_ic1.view(-1,1), R_ic2.view(-1,1)))
        else:
            R_ic1 = torch.zeros(10)
            R_ic2 = torch.zeros(10)

        if switches['bc']:
            # boundary condition residual
            R_bc = wh_coll_hat[self.bc_ids]
        else:
            R_bc = torch.zeros(10)

        return {
            "R_obs" : R_obs.squeeze(),
            "R_pde" : R_pde.squeeze(),
            "R_bc" : R_bc.squeeze(),
            "R_ic" : R_ic.squeeze(),
            "R_cc" : R_cc.squeeze()
        }
    
    def loss_func(self, lambds):

        switches = {}
        for key, value in lambds.items():
            if isinstance(value, list):
                switches[key] = True
            else:
                switches[key] = value>0.0

        residuals = self.calc_residuals(switches)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic = residuals["R_ic"]
        R_cc = residuals["R_cc"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * (torch.mean(R_bc**2))
        L_ic = lambds['ic'] * torch.sum(torch.mean(R_ic**2,dim=0),dim=0)
        L_cc = lambds['cc'][0] * torch.mean(R_cc[:,0]**2) + lambds['cc'][1]*torch.mean(R_cc[:,1]**2)
        loss = L_obs + L_pde + L_cc + L_ic

        return loss, [L_obs, L_pde, L_cc, L_ic]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wd0_col, self.x_col, self.t_col)
        return wp

class osa_pinn_beam_v1(nn.Module):

    def __init__(self, configuration):
        super().__init__()
        self.activation = nn.Tanh

        self.configure(**configuration)

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wdd0, x, t):
        X = torch.cat((w0.view(-1,1), wdd0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y = self.net(X)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(0, self.L, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wdd = config["alphas"]['wdd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wdd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]
        n_x_obs = x_data.shape[0]

        # add zero condition data at boundaries
        w_data_ = torch.cat((torch.zeros(1,n_t_obs),w_data,torch.zeros(1,n_t_obs)),dim=0)
        # w_data_ = torch.cat((w_data,torch.zeros(1,n_t_obs)),dim=0)
        wdd_data_ = torch.cat((torch.zeros(1,n_t_obs),wdd_data,torch.zeros(1,n_t_obs)),dim=0)
        # wdd_data_ = torch.cat((wdd_data,torch.zeros(1,n_t_obs)),dim=0)
        x_data_ = torch.cat((torch.zeros(1,n_t_obs),x_data,self.l*torch.ones(1,n_t_obs)),dim=0)
        # x_data_ = torch.cat((x_data,self.L*torch.ones(1,n_t_obs)),dim=0)
        t_data_ = torch.cat((t_data[:-1,:],t_data[:3,:]),dim=0)

        w_data_1 = w_data_[:-1,1:]  # x=0,t=T
        w_data_2 = w_data_[1:,1:]   # x=L,t=T
        w_data_3 = w_data_[1:,:-1]  # x=L,t=0

        n_t_obs -= 1
        n_x_obs += 1
        # n_x_obs -= 1

        # Observation set in matrix form (uses displacement one time point ahead)
        w0_obs_mat = torch.repeat_interleave(w_data_[:-1,:-1], 3, dim=0)
        wdd0_obs_mat = torch.repeat_interleave(wdd_data_[:-1,:-1], 3, dim=0)
        x_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            x_obs_mat[3*i+1,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
            x_obs_mat[3*i+2,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
        t_obs_mat = torch.cat((
            self.T_hat * torch.ones((2, n_t_obs)),
            torch.zeros((1,n_t_obs))
        ),dim=0).repeat((n_x_obs,1))
        yy_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            yy_obs_mat[3*i,:] = w_data_1[i,:]
            yy_obs_mat[3*i+1,:] = w_data_2[i,:]
            yy_obs_mat[3*i+2,:] = w_data_3[i,:]

        self.w0_obs = w0_obs_mat.reshape(-1).to(self.device)
        self.wdd0_obs = wdd0_obs_mat.reshape(-1).to(self.device)
        self.x_obs = x_obs_mat.reshape(-1).to(self.device)
        self.t_obs = t_obs_mat.reshape(-1).to(self.device)
        self.yy_obs = yy_obs_mat.reshape(-1).to(self.device)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        self.n_col_x = n_x_obs*self.ncx
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x, self.n_col_t))
        wdd0_col = torch.zeros((self.n_col_x, self.n_col_t))
        
        x_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in x
        t_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in time
        x_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true x coordinate of cols)
        t_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true time of cols)

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                wdd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wdd_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x_wind = torch.linspace(0,x_data_[i+1,0]-x_data_[i,0],self.ncx)
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_wind.view(self.ncx,1).repeat(1,self.nct)

                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.view(1,self.nct).repeat(self.ncx,1)

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j] + x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data_[i,j] + self.t_wind.view(1,self.nct).repeat(self.ncx,1)

        self.w0_col = w0_col.reshape(-1).requires_grad_().to(self.device)
        self.wdd0_col = wdd0_col.reshape(-1).requires_grad_().to(self.device)

        self.x_col = x_col.reshape(-1).requires_grad_().to(self.device)
        self.t_col = t_col.reshape(-1).requires_grad_().to(self.device)

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        pred_cat = torch.cat((self.x_pred, self.t_pred), dim=1)
        unqs, counts = torch.unique(pred_cat, return_counts=True, dim=0)
        self.cc_pairs = torch.zeros((sum(counts>1),2), dtype=torch.int32)
        self.cc_pairs = []
        repeated_groups = unqs[counts>1]
        for i, repeated_group in enumerate(repeated_groups):
            repeated_idx = torch.argwhere(torch.all(pred_cat==repeated_group, dim=1)).reshape(-1)
            if (repeated_idx.shape[0]>2):
                repeated_idx_ = torch.combinations(repeated_idx,r=2)
                [self.cc_pairs.append(repeated_idx_[j,:].view(-1)) for j in range(repeated_idx_.shape[0])]
            else:
                self.cc_pairs.append(repeated_idx)

        self.ic_ids = torch.argwhere((self.t_col==torch.tensor(0.0)) * (self.x_col==torch.tensor(0.0)))
        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self, switches):

        if switches['obs']:
            # generate prediction at observation points
            wh_obs_hat = torch.zeros(self.w0_obs.shape[0]).to(self.device)
            wh_obs_hat = self.forward(self.w0_obs, self.wdd0_obs, self.x_obs, self.t_obs).squeeze()
            R_obs = wh_obs_hat - self.yy_obs
        else:
            R_obs = torch.zeros(10).to(self.device)

        # generate prediction over prediction horizon
        wh_coll_hat = torch.zeros(self.w0_col.shape[0]).to(self.device)
        wh_coll_hat[:] = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col).squeeze()

        if switches['pde'] or switches['ic']:

            # retrieve derivatives
            dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
            dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        if switches['pde']:
            dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
            dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
            dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

            # retrieve pde loss parameters
            match self.param_type:
                case "constant":
                    self.m_hat = self.pde_alphas["dt2"]
                    self.k_hat = self.pde_alphas["dx4"]
                case "variable":
                    self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                    self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

            R_cc = torch.zeros((len(self.cc_pairs),2))
            for i in range(len(self.cc_pairs)):
                R_cc[i,0] = (dx4[self.cc_pairs[i][0]] - dx4[self.cc_pairs[i][1]])
                R_cc[i,1] = (dt2[self.cc_pairs[i][0]] - dt2[self.cc_pairs[i][1]])*self.m_hat

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(10)
            R_cc = torch.zeros((10,2))

        if switches['ic']:
            # Initial condition residual
            R_ic1 = self.alpha_wdd * self.wdd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t**2)) * dt2[self.ic_ids]
            R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]
            R_ic = torch.cat((R_ic1.view(-1,1), R_ic2.view(-1,1)))
        else:
            R_ic1 = torch.zeros(10)
            R_ic2 = torch.zeros(10)

        if switches['bc']:
            # boundary condition residual
            R_bc = wh_coll_hat[self.bc_ids]
        else:
            R_bc = torch.zeros(10)

        return {
            "R_obs" : R_obs.squeeze(),
            "R_pde" : R_pde.squeeze(),
            "R_bc" : R_bc.squeeze(),
            "R_ic" : R_ic.squeeze(),
            "R_cc" : R_cc.squeeze()
        }
    
    def loss_func(self, lambds):

        switches = {}
        for key, value in lambds.items():
            if isinstance(value, list):
                switches[key] = True
            else:
                switches[key] = value>0.0

        residuals = self.calc_residuals(switches)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic = residuals["R_ic"]
        R_cc = residuals["R_cc"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * (torch.mean(R_bc**2))
        L_ic = lambds['ic'] * torch.sum(torch.mean(R_ic**2,dim=0),dim=0)
        L_cc = lambds['cc'][0] * torch.mean(R_cc[:,0]**2) + lambds['cc'][1]*torch.mean(R_cc[:,1]**2)
        loss = L_obs + L_pde + L_cc + L_ic

        return loss, [L_obs, L_pde, L_cc, L_ic]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col)
        return wp

class osa_pinn_beam_v1_2(nn.Module):
    """ v1.2 includes the full initial condition at true time 0 """

    def __init__(self, configuration):
        super().__init__()
        self.activation = nn.Tanh

        self.configure(**configuration)

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wdd0, x, t):
        X = torch.cat((w0.view(-1,1), wdd0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y = self.net(X)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(0, self.L, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.true_ic_func = config["ic_func"]  # true initial condition as a function of x

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wdd = config["alphas"]['wdd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wdd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]
        n_x_obs = x_data.shape[0]

        # add zero condition data at boundaries
        w_data_ = torch.cat((torch.zeros(1,n_t_obs),w_data,torch.zeros(1,n_t_obs)),dim=0)
        # w_data_ = torch.cat((w_data,torch.zeros(1,n_t_obs)),dim=0)
        wdd_data_ = torch.cat((torch.zeros(1,n_t_obs),wdd_data,torch.zeros(1,n_t_obs)),dim=0)
        # wdd_data_ = torch.cat((wdd_data,torch.zeros(1,n_t_obs)),dim=0)
        x_data_ = torch.cat((torch.zeros(1,n_t_obs),x_data,self.l*torch.ones(1,n_t_obs)),dim=0)
        # x_data_ = torch.cat((x_data,self.L*torch.ones(1,n_t_obs)),dim=0)
        t_data_ = torch.cat((t_data[:-1,:],t_data[:3,:]),dim=0)

        w_data_1 = w_data_[:-1,1:]  # x=0,t=T
        w_data_2 = w_data_[1:,1:]   # x=L,t=T
        w_data_3 = w_data_[1:,:-1]  # x=L,t=0

        n_t_obs -= 1
        n_x_obs += 1
        # n_x_obs -= 1

        # Observation set in matrix form (uses displacement one time point ahead)
        w0_obs_mat = torch.repeat_interleave(w_data_[:-1,:-1], 3, dim=0)
        wdd0_obs_mat = torch.repeat_interleave(wdd_data_[:-1,:-1], 3, dim=0)
        x_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            x_obs_mat[3*i+1,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
            x_obs_mat[3*i+2,:] = x_data_[i+1,:-1] - x_data_[i,:-1]
        t_obs_mat = torch.cat((
            self.T_hat * torch.ones((2, n_t_obs)),
            torch.zeros((1,n_t_obs))
        ),dim=0).repeat((n_x_obs,1))
        yy_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            yy_obs_mat[3*i,:] = w_data_1[i,:]
            yy_obs_mat[3*i+1,:] = w_data_2[i,:]
            yy_obs_mat[3*i+2,:] = w_data_3[i,:]

        self.w0_obs = w0_obs_mat.reshape(-1).to(self.device)
        self.wdd0_obs = wdd0_obs_mat.reshape(-1).to(self.device)
        self.x_obs = x_obs_mat.reshape(-1).to(self.device)
        self.t_obs = t_obs_mat.reshape(-1).to(self.device)
        self.yy_obs = yy_obs_mat.reshape(-1).to(self.device)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        self.n_col_x = n_x_obs*self.ncx
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x, self.n_col_t))
        wdd0_col = torch.zeros((self.n_col_x, self.n_col_t))
        
        x_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in x
        t_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in time
        x_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true x coordinate of cols)
        t_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true time of cols)

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                wdd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wdd_data_[i,j].item()*torch.ones((self.ncx,self.nct))

                x_wind = torch.linspace(0,x_data_[i+1,0]-x_data_[i,0],self.ncx)
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_wind.view(self.ncx,1).repeat(1,self.nct)

                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.view(1,self.nct).repeat(self.ncx,1)

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j] + x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data_[i,j] + self.t_wind.view(1,self.nct).repeat(self.ncx,1)

        self.w0_col = w0_col.reshape(-1).requires_grad_().to(self.device)
        self.wdd0_col = wdd0_col.reshape(-1).requires_grad_().to(self.device)

        self.x_col = x_col.reshape(-1).requires_grad_().to(self.device)
        self.t_col = t_col.reshape(-1).requires_grad_().to(self.device)

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        pred_cat = torch.cat((self.x_pred, self.t_pred), dim=1)
        unqs, counts = torch.unique(pred_cat, return_counts=True, dim=0)
        self.cc_pairs = torch.zeros((sum(counts>1),2), dtype=torch.int32)
        self.cc_pairs = []
        repeated_groups = unqs[counts>1]
        for i, repeated_group in enumerate(repeated_groups):
            repeated_idx = torch.argwhere(torch.all(pred_cat==repeated_group, dim=1)).reshape(-1)
            if (repeated_idx.shape[0]>2):
                repeated_idx_ = torch.combinations(repeated_idx,r=2)
                [self.cc_pairs.append(repeated_idx_[j,:].view(-1)) for j in range(repeated_idx_.shape[0])]
            else:
                self.cc_pairs.append(repeated_idx)

        self.ic_ids = torch.argwhere((self.t_col==torch.tensor(0.0)) * (self.x_col==torch.tensor(0.0)))

        self.true_ic_ids = torch.argwhere((self.t_pred==torch.tensor(0.0)))[:,0]
        self.true_ic = self.true_ic_func(self.x_pred[self.true_ic_ids]*self.alpha_x)

        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self, switches):

        if switches['obs']:
            # generate prediction at observation points
            wh_obs_hat = torch.zeros(self.w0_obs.shape[0]).to(self.device)
            wh_obs_hat = self.forward(self.w0_obs, self.wdd0_obs, self.x_obs, self.t_obs).squeeze()
            R_obs = wh_obs_hat - self.yy_obs
        else:
            R_obs = torch.zeros(10).to(self.device)

        # generate prediction over prediction horizon
        wh_coll_hat = torch.zeros(self.w0_col.shape[0]).to(self.device)
        wh_coll_hat[:] = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col).squeeze()

        if switches['pde'] or switches['ic']:

            # retrieve derivatives
            dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
            dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        if switches['pde']:
            dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
            dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
            dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

            # retrieve pde loss parameters
            match self.param_type:
                case "constant":
                    self.m_hat = self.pde_alphas["dt2"]
                    self.k_hat = self.pde_alphas["dx4"]
                case "variable":
                    self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                    self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

            R_cc = torch.zeros((len(self.cc_pairs),2))
            for i in range(len(self.cc_pairs)):
                R_cc[i,0] = (dx4[self.cc_pairs[i][0]] - dx4[self.cc_pairs[i][1]])
                R_cc[i,1] = (dt2[self.cc_pairs[i][0]] - dt2[self.cc_pairs[i][1]])*self.m_hat

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(10)
            R_cc = torch.zeros((10,2))

        if switches['ic']:
            # Initial condition residual
            R_ic1 = self.alpha_wdd * self.wdd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t**2)) * dt2[self.ic_ids]
            R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]
            R_ic = torch.cat((R_ic1.view(-1,1), R_ic2.view(-1,1)))
        else:
            R_ic1 = torch.zeros(10)
            R_ic2 = torch.zeros(10)

        if switches['bc']:
            # boundary condition residual
            R_bc = wh_coll_hat[self.bc_ids]
        else:
            R_bc = torch.zeros(10)

        if switches['tic']:
            R_tic = wh_coll_hat[self.true_ic_ids] - self.true_ic.view(-1)
        else:
            R_tic = torch.zeros(10)

        return {
            "R_obs" : R_obs.squeeze(),
            "R_pde" : R_pde.squeeze(),
            "R_bc" : R_bc.squeeze(),
            "R_ic" : R_ic.squeeze(),
            "R_tic" : R_tic.squeeze(),
            "R_cc" : R_cc.squeeze()
        }
    
    def set_loss_switches(self, lambds):

        switches = {}
        for key, value in lambds.items():
            if isinstance(value, list):
                if any([val>0.0 for val in value]):
                    switches[key] = True
                else:
                    switches[key] = False
            else:
                switches[key] = value>0.0
        self.switches = switches
    
    def loss_func(self, lambds):

        residuals = self.calc_residuals(self.switches)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic = residuals["R_ic"]
        R_tic = residuals["R_tic"]
        R_cc = residuals["R_cc"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * (torch.mean(R_bc**2))
        L_ic = lambds['ic'] * torch.sum(torch.mean(R_ic**2,dim=0),dim=0)
        L_tic = lambds['tic'] * (torch.mean(R_tic**2))
        L_cc = lambds['cc'][0] * torch.mean(R_cc[:,0]**2) + lambds['cc'][1]*torch.mean(R_cc[:,1]**2)
        loss = L_obs + L_pde + L_tic + L_ic

        return loss, [L_obs, L_pde, L_tic, L_ic]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col)
        return wp

class osa_pinn_beam_v1_1(nn.Module):
    """ v1.1 add hard boundary conditions using the G and D transformations """

    def __init__(self, configuration):
        super().__init__()
        self.activation = nn.Tanh

        self.configure(**configuration)

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wdd0, x, t):
        X = torch.cat((w0.view(-1,1), wdd0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y = self.net(X)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(0, self.L, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wdd = config["alphas"]['wdd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wdd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]
        n_x_obs = x_data.shape[0]

        # add zero condition data at boundaries
        w_data_ = torch.cat((torch.zeros(1,n_t_obs),w_data,torch.zeros(1,n_t_obs)),dim=0)
        wdd_data_ = torch.cat((torch.zeros(1,n_t_obs),wdd_data,torch.zeros(1,n_t_obs)),dim=0)
        x_data_ = torch.cat((torch.zeros(1,n_t_obs),x_data,self.l*torch.ones(1,n_t_obs)),dim=0)
        t_data_ = torch.cat((t_data,t_data[:2,:]),dim=0)

        w_data_1 = w_data_[:-1,1:]  # x=0,t=T
        w_data_2 = w_data_[1:,1:]   # x=L,t=T
        w_data_3 = w_data_[1:,:-1]  # x=L,t=0

        n_t_obs -= 1
        n_x_obs += 1

        # Observation set in matrix form (uses displacement one time point ahead)
        w0_obs_mat = torch.repeat_interleave(w_data_[:-1,:-1], 3, dim=0)
        wdd0_obs_mat = torch.repeat_interleave(wdd_data_[:-1,:-1], 3, dim=0)
        x_obs_mat = torch.cat((
            torch.zeros(1, n_t_obs),
            self.L * torch.ones(2, n_t_obs)
        ),dim=0).repeat((n_x_obs,1))
        t_obs_mat = torch.cat((
            self.T_hat * torch.ones((2, n_t_obs)),
            torch.zeros((1,n_t_obs))
        ),dim=0).repeat((n_x_obs,1))
        yy_obs_mat = torch.zeros((n_x_obs*3, n_t_obs))
        for i in range(n_x_obs):
            yy_obs_mat[3*i,:] = w_data_1[i,:]
            yy_obs_mat[3*i+1,:] = w_data_2[i,:]
            yy_obs_mat[3*i+2,:] = w_data_3[i,:]

        self.w0_obs = w0_obs_mat.reshape(-1).to(self.device)
        self.wdd0_obs = wdd0_obs_mat.reshape(-1).to(self.device)
        self.x_obs = x_obs_mat.reshape(-1).to(self.device)
        self.t_obs = t_obs_mat.reshape(-1).to(self.device)
        self.yy_obs = yy_obs_mat.reshape(-1).to(self.device)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        self.n_col_x = n_x_obs*self.ncx
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x, self.n_col_t))
        wdd0_col = torch.zeros((self.n_col_x, self.n_col_t))
        
        x_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in x
        t_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in time
        x_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true x coordinate of cols)
        t_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true time of cols)

        G_col = torch.zeros((self.n_col_x, self.n_col_t))

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data_[i,j].item()*torch.ones((self.ncx,self.nct))
                wdd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wdd_data_[i,j].item()*torch.ones((self.ncx,self.nct))
                x_wind = torch.linspace(0,x_data_[i+1,0]-x_data_[i,0],self.ncx)
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.view(1,self.nct).repeat(self.ncx,1)

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data_[i,j] + x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data_[i,j] + self.t_wind.view(1,self.nct).repeat(self.ncx,1)

        D_col = (1-(1-2*self.L/x_col)**(-8)) * (1-(1-2*self.L/(self.L-x_col))**(-8))

        self.w0_col = w0_col.reshape(-1).requires_grad_().to(self.device)
        self.wdd0_col = wdd0_col.reshape(-1).requires_grad_().to(self.device)
        self.x_col = x_col.reshape(-1).requires_grad_().to(self.device)
        self.t_col = t_col.reshape(-1).requires_grad_().to(self.device)
        self.D_col = D_col.reshape(-1).requires_grad_().to(self.device)

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        self.ic_ids = torch.argwhere((self.t_col==torch.tensor(0.0)) * (self.x_col==torch.tensor(0.0)))
        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self, switches):

        if switches['obs']:
            # generate prediction at observation points
            wh_obs_hat = torch.zeros(self.w0_obs.shape[0]).to(self.device)
            wh_obs_hat[:] = self.forward(self.w0_obs, self.wdd0_obs, self.x_obs, self.t_obs).squeeze()
            R_obs = wh_obs_hat - self.yy_obs
        else:
            R_obs = torch.zeros(10).to(self.device)

        # generate prediction over prediction horizon
        wh_coll_hat = torch.zeros(self.w0_col.shape[0]).to(self.device)
        wh_coll_hat[:] = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col).squeeze()

        if switches['pde'] or switches['ic']:

            # retrieve derivatives
            dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
            dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        if switches['pde']:
            dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
            dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
            dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

            # retrieve pde loss parameters
            match self.param_type:
                case "constant":
                    self.m_hat = self.pde_alphas["dt2"]
                    self.k_hat = self.pde_alphas["dx4"]
                case "variable":
                    self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                    self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(10)

        if switches['ic']:
            # Initial condition residual
            R_ic1 = self.alpha_wdd * self.wdd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t**2)) * dt2[self.ic_ids]
            R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]
        else:
            R_ic1 = torch.zeros(10)
            R_ic2 = torch.zeros(10)

        if switches['bc']:
            # boundary condition residual
            R_bc = wh_coll_hat[self.bc_ids]
        else:
            R_bc = torch.zeros(10)

        return {
            "R_obs" : R_obs.squeeze(),
            "R_pde" : R_pde.squeeze(),
            "R_bc" : R_bc.squeeze(),
            "R_ic1" : R_ic1.squeeze(),
            "R_ic2" : R_ic2.squeeze(),
        }
    
    def loss_func(self, lambds):

        switches = {}
        for key, value in lambds.items():
            switches[key] = value>0.0

        residuals = self.calc_residuals(switches)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic1 = residuals["R_ic1"]
        R_ic2 = residuals["R_ic2"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * (torch.mean(R_bc**2))
        L_ic = lambds['ic'] * (torch.mean(R_ic1**2) + torch.mean(R_ic2**2))
        loss = L_obs + L_pde + L_bc + L_ic

        return loss, [L_obs, L_pde, L_bc, L_ic]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col)
        return wp

class osa_pinn_beam_v0_1(nn.Module):
    """ v0.1 had the collocation window go +/-L"""

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wdd0, x, t):
        X = torch.cat((w0.view(-1,1), wdd0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y = self.net(X)
        return y
    
    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.l = config["l"]
        self.L = config["L"]
        self.ncx = config["ncx"]
        self.x_wind = torch.linspace(-self.L, self.L, self.ncx)

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]['t']
        self.alpha_x = config["alphas"]['x']
        self.alpha_w = config["alphas"]['w']
        self.alpha_wdd = config["alphas"]['wdd']
        
        self.alpha_pA = config["alphas"]['pA']
        self.alpha_EI = config["alphas"]['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : self.EI * alpha_dx4 * config["pde_norm_Lambda"]
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * config["pde_norm_Lambda"],
                    "dx4" : alpha_dx4 * config["pde_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, w_data, wdd_data):
        """
        Give all inputs as [nx,nt] matrices
        """

        n_t_obs = x_data.shape[1]-1
        n_x_obs = x_data.shape[0]

        w_data_1 = torch.cat((torch.zeros(1,n_t_obs),w_data[:-1,:-1]), dim=0)  # x=-L,t=0
        w_data_2 = torch.cat((torch.zeros(1,n_t_obs),w_data[:-1,1:]), dim=0)  # x=-L,t=T
        w_data_3 = w_data[:,1:]  # x=0,t=T
        w_data_4 = torch.cat((w_data[1:,1:],torch.zeros(1,n_t_obs)), dim=0)  # x=L,t=T
        w_data_5 = torch.cat((w_data[1:,:-1],torch.zeros(1,n_t_obs)), dim=0)  # x=L,t=0 

        # Observation set in matrix form (uses displacement one time point ahead)
        w0_obs_mat = torch.repeat_interleave(w_data[:,:-1], 5, dim=0)
        wdd0_obs_mat = torch.repeat_interleave(wdd_data[:,:-1], 5, dim=0)
        x_obs_mat = self.L * (torch.cat((
            -1*torch.ones(1, n_t_obs).repeat((2,1)),
            torch.zeros(1, n_t_obs),
            torch.ones(1, n_t_obs).repeat((2,1))
            ),dim=0)).repeat((n_x_obs,1))
        # t_obs_mat = self.T_hat * torch.ones((n_x_obs*5, n_t_obs))
        t_obs_mat = torch.cat((
            torch.zeros((1,n_t_obs)),
            self.T_hat * torch.ones((1, n_t_obs)).repeat(3,1),
            torch.zeros((1,n_t_obs))
        )).repeat((n_x_obs,1))
        yy_obs_mat = torch.zeros((n_x_obs*5, n_t_obs))
        for i in range(n_x_obs):
            yy_obs_mat[5*i,:] = w_data_1[i,:]
            yy_obs_mat[5*i+1,:] = w_data_2[i,:]
            yy_obs_mat[5*i+2,:] = w_data_3[i,:]
            yy_obs_mat[5*i+3,:] = w_data_4[i,:]
            yy_obs_mat[5*i+4,:] = w_data_5[i,:]
        # yy_obs_mat = torch.stack((w_data_bc1,w_data_bc2), dim=2).view(n_x_obs*2,n_t_obs)

        self.w0_obs = w0_obs_mat.reshape(-1,1)
        self.wdd0_obs = wdd0_obs_mat.reshape(-1,1)
        self.x_obs = x_obs_mat.reshape(-1,1)
        self.t_obs = t_obs_mat.reshape(-1,1)
        self.yy_obs = yy_obs_mat.reshape(-1,1)

        # self.w0_obs = w_data[:-1,:-1].reshape(-1,1)
        # self.wdd0_obs = wdd_data[:-1,:-1].reshape(-1,1)
        # self.x_obs = self.L * torch.ones((n_x_obs*2, n_t_obs)).reshape(-1,1)
        # self.t_obs = self.T_hat * torch.ones((n_x_obs*2, n_t_obs)).reshape(-1,1)
        # self.yy_obs = w_data[1:,1:].reshape(-1,1)

        # Collocation set (sets a copy of w0, wdd0 for a vector of time over the time horizon)
        self.n_col_x = n_x_obs*self.ncx
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x, self.n_col_t))
        wdd0_col = torch.zeros((self.n_col_x, self.n_col_t))
        
        x_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in x
        t_col = torch.zeros((self.n_col_x, self.n_col_t))  # collocation points in time
        x_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true x coordinate of cols)
        t_pred = torch.zeros((self.n_col_x, self.n_col_t))  # prediction points (represent true time of cols)

        for i in range(n_x_obs):
            for j in range(n_t_obs):
                w0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = w_data[i,j].item()*torch.ones((self.ncx,self.nct))
                wdd0_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = wdd_data[i,j].item()*torch.ones((self.ncx,self.nct))
                x_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_col[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = self.t_wind.view(1,self.nct).repeat(self.ncx,1)

                # generate a matrix of the time and x for predicted output
                x_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = x_data[i,j] + self.x_wind.view(self.ncx,1).repeat(1,self.nct)
                t_pred[self.ncx*i:self.ncx*(i+1),self.nct*j:self.nct*(j+1)] = t_data[i,j] + self.t_wind.view(1,self.nct).repeat(self.ncx,1)

        self.w0_col = w0_col.reshape(-1,1).requires_grad_()
        self.wdd0_col = wdd0_col.reshape(-1,1).requires_grad_()
        self.x_col = x_col.reshape(-1,1).requires_grad_()
        self.t_col = t_col.reshape(-1,1).requires_grad_()

        self.x_pred = x_pred.reshape(-1,1)
        self.t_pred = t_pred.reshape(-1,1)

        self.ic_ids = torch.argwhere((self.t_col[:,0]==torch.tensor(0.0)) * (self.x_col[:,0]==torch.tensor(0.0)))
        bc_ids1 = torch.argwhere(self.x_pred[:,0]==torch.tensor(0.0))
        bc_ids2 = torch.argwhere(self.x_pred[:,0]==torch.tensor(self.l))
        self.bc_ids = torch.unique(torch.cat((bc_ids1,bc_ids2),dim=0))

        return x_pred, t_pred
    
    def calc_residuals(self, switches):

        if switches['obs']:
            # generate prediction at observation points
            wh_obs_hat = self.forward(self.w0_obs, self.wdd0_obs, self.x_obs, self.t_obs)
            R_obs = wh_obs_hat - self.yy_obs
        else:
            R_obs = torch.zeros(10)

        # generate prediction over prediction horizon
        wh_coll_hat = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col)

        if switches['pde'] or switches['ic']:

            # retrieve derivatives
            dt = torch.autograd.grad(wh_coll_hat, self.t_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
            dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        if switches['pde']:
            dx = torch.autograd.grad(wh_coll_hat, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
            dx2 = torch.autograd.grad(dx, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
            dx3 = torch.autograd.grad(dx2, self.x_col, torch.ones_like(wh_coll_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, self.x_col, torch.ones_like(wh_coll_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

            # retrieve pde loss parameters
            match self.param_type:
                case "constant":
                    self.m_hat = self.pde_alphas["dt2"]
                    self.k_hat = self.pde_alphas["dx4"]
                case "variable":
                    self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                    self.k_hat = self.pde_alphas["dx4"] * self.phys_params[1]

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(10)

        if switches['ic']:
            # Initial condition residual
            R_ic1 = self.alpha_wdd * self.wdd0_col[self.ic_ids] - (self.alpha_w/(self.alpha_t**2)) * dt2[self.ic_ids]
            R_ic2 = self.w0_col[self.ic_ids] - wh_coll_hat[self.ic_ids]
        else:
            R_ic1 = torch.zeros(10)
            R_ic2 = torch.zeros(10)

        if switches['bc']:
            # boundary condition residual
            R_bc = wh_coll_hat[self.bc_ids]
        else:
            R_bc = torch.zeros(10)

        return {
            "R_obs" : R_obs.squeeze(),
            "R_pde" : R_pde.squeeze(),
            "R_bc" : R_bc.squeeze(),
            "R_ic1" : R_ic1.squeeze(),
            "R_ic2" : R_ic2.squeeze(),
        }
    
    def loss_func(self, lambds):

        switches = {}
        for key, value in lambds.items():
            switches[key] = value>0.0

        residuals = self.calc_residuals(switches)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc = residuals["R_bc"]
        R_ic1 = residuals["R_ic1"]
        R_ic2 = residuals["R_ic2"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * (torch.mean(R_bc**2))
        L_ic = lambds['ic'] * (torch.mean(R_ic1**2) + torch.mean(R_ic2**2))
        loss = L_obs + L_pde + L_bc + L_ic

        return loss, [L_obs, L_pde, L_bc, L_ic]
    
    def predict(self):
        wp = self.forward(self.w0_col, self.wdd0_col, self.x_col, self.t_col)
        return wp


class bbnn(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

        self.build_net()
    
    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
        
    def forward(self, x):
        x = self.net(x)
        return x

    def predict(self, tp):
        yp = self.forward(tp)
        return yp

    def loss_func(self, x_obs, y_obs):
        yp_obs = self.forward(x_obs)
        loss = torch.mean((yp_obs - y_obs)**2)
        return loss


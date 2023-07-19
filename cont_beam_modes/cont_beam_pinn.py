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
    

class beam_ss_pinn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_modes = config["n_modes"]
        self.n_sens = config["n_sens"]

        if config["forcing"] == None:
            self.n_input = 2*self.n_sens + 1
        self.n_output = 2*self.n_modes

        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.activation = nn.Tanh

        self.build_net()
        self.configure(**config)

        self.n_col_t = config['nct']

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net

    def forward(self, w0, wdot0, t, f0=None):
        if f0 is None:
            x = torch.cat((w0, wdot0, t.view(-1,1)), dim=1)
        else:
            x = torch.cat((w0, wdot0, t.view(-1,1), f0), dim=1)
        tau_p = self.net(x)
        return tau_p
    
    def configure(self, **config):

        self.config = config

        self.param_type = config['phys_params']['par_type']
        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match self.param_type:
            case 'constant':
                self.M = config['phys_params']['M']
                self.K = config['phys_params']['K']
                self.C = config['phys_params']['C']
                self.A = torch.cat((
                    torch.cat((torch.zeros((self.n_modes,self.n_modes)),torch.eye(self.n_modes)), dim=1),
                    torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
                    ), dim=0)
                self.H = torch.cat((torch.zeros((self.n_modes,self.n_modes)),torch.linalg.inv(self.M)), dim=0)
            case 'variable':
                self.register_parameter('phys_params', nn.Parameter(torch.tensor([config['phys_params']['M'],config['phys_params']['K']])))
        
        self.phis = config['modes']['phis']  #[x,mode]
        self.xx = config['modes']['xx']
        self.n_pred_x = self.phis.shape[0]
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config['alphas']['t']
        self.alpha_x = config['alphas']['x']
        self.alpha_w = config['alphas']['w']
        self.alpha_wdot = config['alphas']['wdot']
        self.alpha_X = torch.cat((self.alpha_w*torch.ones((self.n_modes,1)),self.alpha_wdot*torch.ones((self.n_modes,1))), dim=0)

        self.alpha_M = config['alphas']['M']
        self.alpha_K = config['alphas']['K']
    
    def set_colls_and_obs(self, phi_obs, t_data, w_data, wdot_data):

        # phi_obs -> [n_sens, n_modes]
        # t_data -> [samples]
        # w_data/wdot_data -> [samples, n_sens]
        n_obs = w_data.shape[0]-1

        # observation set (uses state one time step ahead)
        self.w0_obs = w_data[:-1,:]
        self.wdot0_obs = wdot_data[:-1,:]
        self.t_obs = torch.zeros((n_obs,1))
        for i in range(n_obs):
            self.t_obs[i] = t_data[i+1] - t_data[i]  # time at the end of the horizon (window)
        self.yy_obs = torch.cat((w_data[1:,:], wdot_data[1:,:]), dim=1).requires_grad_()  # state at the end of the horizon
        self.phi_obs = phi_obs.requires_grad_()  # [nx, nm]

        # collocation set (sets a copy of state vector over the time horizon)
        self.n_col = n_obs*self.n_col_t
        w0_col = torch.zeros((self.n_col, self.n_sens))  # [n_col, n_sens]
        wdot0_col = torch.zeros((self.n_col, self.n_sens))  # [n_col, n_sens]
        t_col = torch.zeros((self.n_col, 1))  # [n_col]
        t_pred = torch.zeros((self.n_col, 1))  # [n_col]
        for i in range(n_obs):
            for j in range(self.n_sens):
                w0_col[self.n_col_t*i:self.n_col_t*(i+1),j] = w_data[i,j]*torch.ones(self.n_col_t)
                wdot0_col[self.n_col_t*i:self.n_col_t*(i+1),j] = wdot_data[i,j]*torch.ones(self.n_col_t)
            t_col[self.n_col_t*i:self.n_col_t*(i+1),0] = torch.linspace(0, t_data[i+1].item()-t_data[i].item(), self.n_col_t)

            # generates a vector of the time for the predicted output, by simply adding the total window onto the current time in the data
            t_pred[self.n_col_t*i:self.n_col_t*(i+1),0] = t_data[i] + torch.linspace(0, t_data[i+1].item()-t_data[i].item(), self.n_col_t)

        self.w0_col = w0_col.requires_grad_()
        self.wdot0_col = wdot0_col.requires_grad_()
        self.t_col = t_col.requires_grad_()

        self.ic_ids = torch.argwhere(t_col[:,0]==torch.tensor(0.0)).view(-1)

        return t_pred.view(-1)

    def calc_residuals(self, switches):

        # prediction and residual over observed data
        if switches['obs']:
            tau_hat_p_obs = self.forward(self.w0_obs, self.wdot0_obs, self.t_obs)  # [samples, 2n_modes]
            wh_obs_mat = torch.zeros((tau_hat_p_obs.shape[0], 2*self.n_sens, self.n_modes))  # [samples, 2n_sens, n_modes]
            for i in range(self.n_sens):
                for j in range(self.n_modes):
                    wh_obs_mat[:,i,j] = self.phi_obs[i,j] * tau_hat_p_obs[:,j]
                    wh_obs_mat[:,self.n_sens+i,j] = self.phi_obs[i,j] * tau_hat_p_obs[:,self.n_modes+j]
            wh_obs_hat = torch.sum(wh_obs_mat, dim=2)
            # wh_obs_hat = torch.sum((qh_obs_hat * self.phi_obs), dim=1)
            R_obs = torch.sqrt(torch.sum((wh_obs_hat - self.yy_obs)**2,dim=1))
        else:
            R_obs = torch.zeros((10,10))

        # prediction and physics residual from collocation points
        if switches['pde']:
            tau_hat_p_coll = self.forward(self.w0_col, self.wdot0_col, self.t_col)  # N_tau(t)  [samples, 2n_modes]
            wh_col_mat = torch.zeros((tau_hat_p_coll.shape[0], 2*self.n_sens, self.n_modes))
            for i in range(self.n_sens):
                for j in range(self.n_modes):
                    wh_col_mat[:,i,j] = self.phi_obs[i,j] * tau_hat_p_coll[:,j]
                    wh_col_mat[:,self.n_sens+i,j] = self.phi_obs[i,j] * tau_hat_p_coll[:,self.n_modes+j]
            wh_col_ = torch.sum(wh_col_mat, dim=2)

            # retrieve derivatives
            dtau_dt = torch.zeros_like(tau_hat_p_coll)
            for i in range(tau_hat_p_coll.shape[1]):
                dtau_dt[:,i] = torch.autograd.grad(tau_hat_p_coll[:,i], self.t_col, torch.ones_like(tau_hat_p_coll[:,i]), create_graph=True)[0][:,0]  # ∂_t-hat N_tau-hat
            R_ = (self.alpha_X/self.alpha_t)*dtau_dt.T - self.A@(self.alpha_X*tau_hat_p_coll.T)
            R_pde = R_[self.n_modes:,:].T
            R_cc = R_[:self.n_modes,:].T
        else:
            R_pde = torch.zeros((10,10))
            R_cc = torch.zeros((10,10))

        if switches['ic']:
            R_ic1 = self.alpha_w * self.w0_col[self.ic_ids,:] - self.alpha_w * wh_col_[self.ic_ids,:self.n_sens]
            R_ic2 = self.alpha_wdot * self.wdot0_col[self.ic_ids,:] - self.alpha_wdot * wh_col_[self.ic_ids,self.n_sens:]
            R_ic = torch.cat((R_ic1.squeeze(), R_ic2.squeeze()), dim=1)
        else:
            R_ic = torch.zeros((10,10))

        return {
            'R_obs' : R_obs,
            'R_ic' : R_ic,
            'R_cc' : R_cc,
            'R_pde' : R_pde
        }
    
    def loss_func(self, lambdas):

        residuals = self.calc_residuals(self.switches)
        R_obs = residuals["R_obs"]
        R_ic = residuals["R_ic"]
        R_cc = residuals["R_cc"]
        R_pde = residuals["R_pde"]

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_ic = lambdas['ic'] * torch.sum(torch.mean(R_ic**2, dim=0), dim=0)/self.n_sens
        L_cc = lambdas['cc'] * torch.sum(torch.mean(R_cc**2, dim=0), dim=0)
        L_pde = lambdas['pde'] * torch.sum(torch.mean(R_pde**2, dim=0), dim=0)
        loss = L_obs + L_ic + L_cc + L_pde

        return loss, [L_obs, L_ic, L_cc, L_pde]
    
    def set_loss_switches(self, lambdas):
        switches = {}
        for key, value in lambdas.items():
            switches[key] = value>0.0
        self.switches = switches
    
    def predict(self):
        tau_pred = self.forward(self.w0_col, self.wdot0_col, self.t_col)
        w_pred_mat_ = torch.zeros((tau_pred.shape[0], 2*self.n_pred_x, self.n_modes))
        for i in range(self.n_pred_x):
            for j in range(self.n_modes):
                w_pred_mat_[:,i,j] = self.phis[i,j] * tau_pred[:,j]
                w_pred_mat_[:,self.n_sens+i,j] = self.phis[i,j] * tau_pred[:,self.n_modes+j]
        w_pred_mat = torch.sum(w_pred_mat_, dim=2)
        return w_pred_mat



class ann(nn.Module):

    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_layers = n_layers
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

class beam_mode_pinn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.n_modes = config["n_modes"]
        self.activation = nn.Tanh

        self.build_nets()
        self.configure(**config)

        self.n_col_t = config['nct']

    def build_nets(self):
        self.nets = []
        self.parameters = []
        for n in range(self.n_modes):
            self.nets.append(ann(self.n_input, self.n_output, self.n_hidden, self.n_layers))
            self.parameters = self.parameters + list(self.nets[n].parameters())
    
    def configure(self, **config):

        self.config = config

        self.param_type = config['phys_params']['par_type']
        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match self.param_type:
            case 'constant':
                self.EI = config['phys_params']['EI']
                self.pA = config['phys_params']['pA']
            case 'variable':
                self.register_parameter('phys_params', nn.Parameter(torch.tensor([config['phys_params']['EI'],config['phys_params']['pA']])))
        
        self.phis = config['modes']['phis']  #[x,mode]
        self.phis_dx4 = config['modes']['phi_dx4']  #[x,mode]
        self.xx = config['modes']['xx']
        self.n_col_x = self.phis.shape[0]
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config['alphas']['t']
        self.alpha_x = config['alphas']['x']
        self.alpha_w = config['alphas']['w']

        self.alpha_EI = config['alphas']['EI']
        self.alpha_pA = config['alphas']['pA']

        alpha_dt2 = 1.0/(self.alpha_t**2)
        alpha_dx4 = 1.0/(self.alpha_x**4)

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

    def forward(self, t):
        q_n = torch.zeros((t.shape[0],self.n_modes))
        for n in range(self.n_modes):
            q_n[:,n] = self.nets[n](t.view(-1,1)).view(-1)
        return q_n
    
    def set_colls_and_obs(self, phi_obs, t_vec, w_data, t_max):

        # data in matrix format as [nx, nt]
        nx_obs = w_data.shape[0]
        nt_obs = w_data.shape[1]
        self.phi_obs = phi_obs.requires_grad_()  # [nx, nm]
        self.t_obs = t_vec.view(-1,1)
        self.w_obs = w_data

        # generate vector of collocation points
        t_col_vec = torch.linspace(0, t_max, self.n_col_t)
        self.t_col = t_col_vec.requires_grad_()
        self.phi_col = self.phis
        self.phi_col_dx4 = self.phis_dx4
    
    # def set_colls_and_obs(self, phi_data, t_data, w_data, t_max):

    #     # data in matrix format as [nx, nt]
    #     nx = w_data.shape[0]
    #     nt = w_data.shape[1]
    #     # phi_data in format [nx, nt, nm]
    #     phi_obs = torch.zeros((nx*nt, self.n_modes))
    #     for n in range(self.n_modes):
    #         phi_obs[:,n] = phi_data[:,:,n].reshape(-1)
    #     self.phi_obs = phi_obs.requires_grad_()
    #     self.t_obs = t_data.reshape(-1)
    #     self.w_obs = w_data.reshape(-1)

    #     # generate matrix versions of collocation points using meshgrid
    #     self.n_col = self.n_col_x*self.n_col_t
    #     t_col_vec = torch.linspace(0, t_max, self.n_col_t)
    #     phi_col = torch.zeros((self.n_col,self.n_modes))
    #     phi_col_dx4 = torch.zeros((self.n_col,self.n_modes))
    #     for n in range(self.n_modes):
    #         phi_col[:,n] = self.phis[:,n].repeat(1,self.n_col_t).reshape(-1)
    #         phi_col_dx4[:,n] = self.phis_dx4[:,n].repeat(1,self.n_col_t).reshape(-1)
    #     t_col_mat = t_col_vec.repeat(self.n_col_x,1)
    #     self.t_col = t_col_mat.reshape(-1).requires_grad_()
    #     self.phi_col = phi_col
    #     self.phi_col_dx4 = phi_col_dx4

    
    def calc_residuals(self, switches):

        # prediction and residual over observed data
        if switches['obs']:
            qh_obs_hat = self.forward(self.t_obs)
            wh_obs_mat = torch.zeros((self.phi_obs.shape[0], qh_obs_hat.shape[0], self.n_modes))
            for n in range(self.n_modes):
                wh_obs_mat[:,:,n] = self.phi_obs[:,n].view(-1,1) @ qh_obs_hat[:,n].view(-1,1).T
            wh_obs_hat = torch.sum(wh_obs_mat, dim=2)
            # wh_obs_hat = torch.sum((qh_obs_hat * self.phi_obs), dim=1)
            R_obs = wh_obs_hat - self.w_obs
        else:
            R_obs = torch.zeros((10,10))

        # retrieve pde loss parameters
        match self.param_type:
            case 'constant':
                self.EI_hat = self.pde_alphas['dx4']
                self.pA_hat = self.pde_alphas['dt2']
            case 'variable':
                self.EI_hat = self.pde_alphas['dx4'] * self.phys_params[0] * self.alpha_EI
                self.pA_hat = self.pde_alphas['dt2'] * self.phys_params[1] * self.alpha_pA

        # prediction and physics residual from collocation points
        if switches['pde']:
            qh_coll_hat = self.forward(self.t_col)  # N_q(t)
            # retrieve derivatives
            dt = torch.zeros((self.n_col_t, self.n_modes))
            dt2 = torch.zeros((self.n_col_t, self.n_modes))
            R_pde_1 = torch.zeros((self.n_col_x, self.n_col_t, self.n_modes))
            R_pde_2 = torch.zeros((self.n_col_x, self.n_col_t, self.n_modes))
            for n in range(self.n_modes):
                dt[:,n] = torch.autograd.grad(qh_coll_hat[:,n], self.t_col, torch.ones_like(qh_coll_hat[:,n]), create_graph=True)[0]  # ∂_t-hat N_q-hat
                dt2[:,n] = torch.autograd.grad(dt[:,n], self.t_col, torch.ones_like(dt[:,n]), create_graph=True)[0]  # ∂^2_t-hat N_q-hat
                R_pde_1[:,:,n] = self.phi_col_dx4[:,n].view(-1,1) @ qh_coll_hat[:,n].view(-1,1).T
                R_pde_2[:,:,n] = self.phi_col[:,n].view(-1,1) @ dt2[:,n].view(-1,1).T
            
            # calculate pde residual
            R_pde = self.EI_hat * torch.sum(R_pde_1, dim=2) + self.pA_hat * torch.sum(R_pde_2, dim=2)
        else:
            R_pde = torch.zeros((10,10))

        return {
            'R_obs' : R_obs,
            'R_pde' : R_pde
        }
    
    def loss_func(self, lambds):

        residuals = self.calc_residuals(self.switches)
        R_obs = residuals['R_obs']
        R_pde = residuals['R_pde']

        L_obs = lambds['obs'] * torch.mean((R_obs**2).reshape(-1))
        L_pde = lambds['pde'] * torch.mean((R_pde**2).reshape(-1))
        loss = L_obs + L_pde

        return loss, [L_obs, L_pde]
    
    def set_loss_switches(self, lambdas):

        switches = {}
        for key, value in lambdas.items():
            switches[key] = value>0.0
        self.switches = switches
    
    def predict(self):
        q_pred = self.forward(self.t_col)
        w_pred_mat_ = torch.zeros((self.n_col_x, self.n_col_t, self.n_modes))
        for n in range(self.n_modes):
            w_pred_mat_[:,:,n] = self.phi_col[:,n].view(-1,1) @ q_pred[:,n].view(-1,1).T
        w_pred_mat = torch.sum(w_pred_mat_, dim=2)
        return w_pred_mat

class beam_third_mode_pinn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.activation = nn.Tanh

        self.build_net()
        self.configure(**config)

        self.n_col_t = config['nct']

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def configure(self, **config):

        self.config = config

        self.param_type = config['phys_params']['par_type']
        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match self.param_type:
            case 'constant':
                self.EI = config['phys_params']['EI']
                self.pA = config['phys_params']['pA']
            case 'variable':
                self.register_parameter('phys_params', nn.Parameter(torch.tensor([config['phys_params']['EI'],config['phys_params']['pA']])))
        
        self.phis = config['modes']['phis']  #[x,mode]
        self.phis_dx4 = config['modes']['phi_dx4']  #[x,mode]
        self.xx = config['modes']['xx']
        self.n_col_x = self.phis.shape[0]
    
    def set_norm_params(self):
        config = self.config
        self.alpha_t = config['alphas']['t']
        self.alpha_x = config['alphas']['x']
        self.alpha_w = config['alphas']['w']

        self.alpha_EI = config['alphas']['EI']
        self.alpha_pA = config['alphas']['pA']

        alpha_dt2 = 1.0/(self.alpha_t**2)
        alpha_dx4 = 1.0/(self.alpha_x**4)

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

    def forward(self, t, G=0.0, D=1.0):
        y_ = self.net(t.view(-1,1))
        qp = G + D * y_.view(-1)  # prediction of q (N_q)
        return qp
    
    def set_colls_and_obs(self, phi_data, t_data, w_data, t_max):

        # data in matrix format as [nx, nt]
        self.phi_obs = phi_data.reshape(-1).requires_grad_()
        self.t_obs = t_data.reshape(-1)
        self.w_obs = w_data.reshape(-1)

        # generate matrix versions of collocation points using meshgrid
        t_col_vec = torch.linspace(0, t_max, self.n_col_t)
        phi_col_mat, t_col_mat = torch.meshgrid((self.phis[:,2], t_col_vec), indexing="ij")
        phi_dx4_mat, t_col_mat = torch.meshgrid((self.phis_dx4[:,2], t_col_vec), indexing="ij")
        self.t_col = t_col_mat.reshape(-1).requires_grad_()
        self.phi_col = phi_col_mat.reshape(-1)
        self.phi_col_dx4 = phi_dx4_mat.reshape(-1)

    
    def calc_residuals(self):

        # prediction and residual over observed data
        qh_obs_hat = self.forward(self.t_obs)
        wh_obs_hat = qh_obs_hat * self.phi_obs
        R_obs = wh_obs_hat - self.w_obs

        # prediction and physics residual from collocation points
        qh_coll_hat = self.forward(self.t_col)  # N_q(t)
        # retrieve derivatives
        dt = torch.autograd.grad(qh_coll_hat, self.t_col, torch.ones_like(qh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_q-hat
        dt2 = torch.autograd.grad(dt, self.t_col, torch.ones_like(dt), create_graph=True)[0]  # ∂^2_t-hat N_q-hat

        # retrieve pde loss parameters
        match self.param_type:
            case 'constant':
                self.EI_hat = self.pde_alphas['dx4']
                self.pA_hat = self.pde_alphas['dt2']
            case 'variable':
                self.EI_hat = self.pde_alphas['dx4'] * self.phys_params[0] * self.alpha_EI
                self.pA_hat = self.pde_alphas['dt2'] * self.phys_params[1] * self.alpha_pA
        
        # calculate pde residual
        R_pde = self.EI_hat * self.phi_col_dx4 * qh_coll_hat + self.pA_hat * self.phi_col * dt2

        return {
            'R_obs' : R_obs,
            'R_pde' : R_pde
        }
    
    def loss_func(self, lambds):

        residuals = self.calc_residuals()
        R_obs = residuals['R_obs']
        R_pde = residuals['R_pde']

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        loss = L_obs + L_pde

        return loss, [L_obs, L_pde]
    
    def predict(self):
        q_pred = self.forward(self.t_col)
        w_pred = q_pred * self.phi_col
        return w_pred



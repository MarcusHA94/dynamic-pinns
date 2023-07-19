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

    def __init__(self, configuration):
        super().__init__()
        self.activation = nn.Tanh

        self.configure(**configuration)

        self.build_net()

    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.l = config["l"]
        self.ncx = config["ncx"]

        self.T = config["T"]
        self.T_hat = config["T_hat"]
        self.nct = config["nct"]
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)

        self.true_ic_func = config["ic_func"]  # true initial condition as a function of x

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, w0, wd0, x, t, G=0.0, D=1.0):
        X = torch.cat((w0.view(-1,1), wd0.view(-1,1), x.view(-1,1), t.view(-1,1)), dim=1)
        y_ = self.net(X)
        y = G + D * y_.view(-1)
        return y
    
    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"}}:
                self.pA = config["phys_params"]['pA']
                self.EI = config["phys_params"]['EI']
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]['pA'], config["phys_params"]['EI']])))
        self.phis = config['modes']['phis']  #[x,mode]
        self.xx = config['modes']['xx']
        self.n_col_x = self.phis.shape[0]

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

        n_x_obs = w_data.shape[0]
        n_t_obs = w_data.shape[1]-1

        # observation set in matrix form (uses displacement one data point ahead)
        w0_obs_mat = w_data[:,:-1]    # initial displacement input
        wd0_obs_mat = wd_data[:,:-1]  # initial velocity input
        x_obs_mat = x_data[:,:-1]
        t_obs_mat = torch.zeros((n_x_obs,n_t_obs))
        for i in range(n_t_obs):
            t_obs_mat[:,i] = t_data[:,i+1] - t_data[:,i]
        yy_obs_mat = w_data[:,1:]

        self.w0_obs = w0_obs_mat.reshape(-1).requires_grad_()
        self.wd0_obs = wd0_obs_mat.reshape(-1).requires_grad_()
        self.x_obs = x_obs_mat.reshape(-1).requires_grad_()
        self.t_obs = t_obs_mat.reshape(-1).requires_grad_()
        self.yy_obs = yy_obs_mat.reshape(-1).requires_grad_()

        # collocation set (sets a copy of w0, wd0 for a vector of time over the time horizon)
        self.n_col_t = n_t_obs*self.nct
        w0_col = torch.zeros((self.n_col_x,self.n_col_t))
        wd0_col = torch.zeros((self.n_col_x,self.n_col_t))
        x_col = torch.zeros((self.n_col_x,self.n_col_t))
        t_col = torch.zeros((self.n_col_x,self.n_col_t))

        t_pred = torch.zeros((self.n_col_x,self.n_col_t))

        for i in range(n_t_obs):

            w0_col[:,self.nct*i:self.nct*(i+1)] = w_data[:,i]


        
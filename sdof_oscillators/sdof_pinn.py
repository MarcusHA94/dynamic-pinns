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
        else:
            dmax = max_mag_data(data)
        data_norm = data/dmax
        return data_norm, dmax

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

class sdof_pinn(nn.Module):

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

    def configure(self, **config):

        self.config = config

        self.nonlinearity = config["nonlinearity"]
        self.forcing = config["forcing"]
        self.param_type = config["phys_params"]["par_type"]

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"linear"}:
                self.k = config["phys_params"]['k']
                self.c = config["phys_params"]['c']
                self.phys_params = torch.tensor([self.c, self.k])
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"cubic"}:
                self.k = config["phys_params"]['k']
                self.c = config["phys_params"]['c']
                self.k3 = config["phys_params"]['k3']
                self.phys_params = torch.tensor([self.c, self.k, self.k3])
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"linear"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"]])))
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"cubic"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"], config["phys_params"]["k3"]])))
        match config["forcing"]:
            case dict():
                self.force = torch.tensor(config["forcing"]["F_hat"]).reshape(-1,1)

    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        if config["forcing"] != None:
            self.alpha_F = config["alphas"]["F"]
        
        match config:
            case {"nonlinearity":"linear","forcing":None}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"linear","forcing":dict()}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "ff" : alpha_ff * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"cubic","forcing":None}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                self.alpha_k3 = config["alphas"]["k3"]
                
                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d0_3" : alpha_d0_3 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"cubic","forcing":dict()}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                self.alpha_k3 = config["alphas"]["k3"]
                
                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d0_3" : alpha_d0_3 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "ff" : alpha_ff * config["ode_norm_Lambda"]
                }

    def calc_residuals(self, t_pde_hat, t_obs, x_obs):

        # observation residual
        xh_obs = self.forward(t_obs)  # N_y-hat or N_y (in Ω_a)
        R_obs = xh_obs - x_obs

        # ode values
        xh_pde_hat = self.forward(t_pde_hat)   # N_y-hat (in Ω_ode)
        dx = torch.autograd.grad(xh_pde_hat, t_pde_hat, torch.ones_like(xh_pde_hat), create_graph=True)[0]  # ∂_t-hat N_y-hat
        dx2 = torch.autograd.grad(dx, t_pde_hat, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_y-hat

        # retrieve ode loss parameters
        self.m_hat = self.ode_alphas["d2"]
        match self.param_type:
            case "constant":
                self.c_hat = self.ode_alphas["d1"] * self.c
                self.k_hat = self.ode_alphas["d0"] * self.k
                match self.config["nonlinearity"]:
                    case "cubic":
                        self.k3_hat = self.ode_alphas["d0_3"] * self.k3
            case "variable":
                self.c_hat = self.ode_alphas["d1"] * self.phys_params[0] * self.alpha_c
                self.k_hat = self.ode_alphas["d0"] * self.phys_params[1] * self.alpha_k
                match self.config["nonlinearity"]:
                    case "cubic":
                        self.k3_hat = self.ode_alphas["d0_3"] * self.phys_params[2] * self.alpha_k3
        match self.config["forcing"]:
            case dict():
                self.eta = self.ode_alphas["ff"]

        # calculate ode residual
        match self.config:
            case {"nonlinearity":"linear","forcing":None}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat
            case {"nonlinearity":"cubic","forcing":None}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat + self.k3_hat * xh_pde_hat**3
            case {"nonlinearity":"linear","forcing":{}}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat - self.eta * self.force
            case {"nonlinearity":"cubic","forcing":{}}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat + self.k3_hat * xh_pde_hat**3 - self.eta * self.force

        return {
            "R_obs" : R_obs,
            "R_ode" : R_ode
        }

    def loss_func(self, t_pde, t_obs, x_obs, lambdas):
        residuals = self.calc_residuals(t_pde, t_obs, x_obs)
        R_obs = residuals["R_obs"]
        R_ode = residuals["R_ode"]

        L_obs = lambdas['obs'].item() * torch.mean(R_obs**2)
        L_ode = lambdas['ode'].item() * torch.mean(R_ode**2)
        loss = L_obs + L_ode

        return loss, [L_obs, L_ode]

    def predict(self, tp):
        yp = self.forward(tp)
        return yp

class sdof_free_pinn(nn.Module):
    
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
    
    def nonlinearity(self, nonlin_style="lin_osc"):
        self.nonlin_style = nonlin_style

    def set_phys_params(self, params, par_type):
        self.param_type = par_type
        match par_type:
            case "constant":
                match self.nonlin_style:
                    case "lin_osc":
                        self.k = params['k']
                        self.c = params['c']
                        self.phys_params = torch.tensor([self.c, self.k])
                    case "cubic_stiffness":
                        self.k = params['k']
                        self.k3 = params['k3']
                        self.c = params['c']
                        self.phys_params = torch.tensor([self.c, self.k, self.k3])
            case "variable":
                match self.nonlin_style:
                    case "lin_osc":
                        self.register_parameter("phys_params", nn.Parameter(torch.tensor([params["c"], params["k"]])))
                    case "cubic_stiffness":
                        self.register_parameter("phys_params", nn.Parameter(torch.tensor([params["c"], params["k"], params["k3"]])))

    def set_norm_params(self, alphas, ode_norm_Lambda):
        self.alpha_t = alphas["t"]
        self.alpha_x = alphas["x"]

        match self.nonlin_style:
            case "lin_osc":
                self.alpha_k = alphas["k"]
                self.alpha_c = alphas["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0 * ode_norm_Lambda,
                    "d1" : alpha_d1 * ode_norm_Lambda,
                    "d2" : alpha_d2 * ode_norm_Lambda
                }
                
            case "cubic_stiffness":
                self.alpha_k = alphas["k"]
                self.alpha_k3 = alphas["k3"]
                self.alpha_c = alphas["c"]

                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0 * ode_norm_Lambda,
                    "d0_3" : alpha_d0_3 * ode_norm_Lambda,
                    "d1" : alpha_d1 * ode_norm_Lambda,
                    "d2" : alpha_d2 * ode_norm_Lambda
                }
        
    def forward(self, x):
        x = self.net(x)
        return x

    def calc_residuals(self, t_pde_hat, t_obs, x_obs):

        match self.param_type:
            case "constant":
                self.m_hat = self.ode_alphas["d2"]
                self.c_hat = self.ode_alphas["d1"] * self.c
                self.k_hat = self.ode_alphas["d0"] * self.k
                if self.nonlin_style == "cubic_stiffness":
                    self.k3_hat = self.ode_alphas["d0_3"] * self.k3

            case "variable":
                self.m_hat = self.ode_alphas["d2"]
                self.c_hat = self.ode_alphas["d1"] * self.phys_params[0] * self.alpha_c
                self.k_hat = self.ode_alphas["d0"] * self.phys_params[1] * self.alpha_k
                if self.nonlin_style == "cubic_stiffness":
                    self.k3_hat = self.ode_alphas["d0_3"] * self.phys_params[2] * self.alpha_k3

        # observation loss
        xh_obs = self.forward(t_obs)  # N_y-hat or N_y (in Ω_a)
        R_obs = xh_obs - x_obs

        # ode residual
        xh_pde_hat = self.forward(t_pde_hat)   # N_y-hat (in Ω_ode)
        dx = torch.autograd.grad(xh_pde_hat, t_pde_hat, torch.ones_like(xh_pde_hat), create_graph=True)[0]  # ∂_t-hat N_y-hat
        dx2 = torch.autograd.grad(dx, t_pde_hat, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_y-hat
        match self.nonlin_style:
            case "lin_osc":
                R_pde = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat
            case "cubic_stiffness":
                R_pde = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat + self.k3_hat * xh_pde_hat**3

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde
        }

    def loss_func(self, t_pde, t_obs, x_obs, lambds):
        residuals = self.calc_residuals(t_pde, t_obs, x_obs)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]

        L_obs = lambds[0] * torch.mean(R_obs**2)
        L_pde = lambds[1] * torch.mean(R_pde**2)
        loss = L_obs + L_pde

        return loss, [L_obs, L_pde]

    def predict(self, tp):
        yp = self.forward(tp)
        return yp

class sdof_forced_pinn(nn.Module):
    
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
    
    def nonlinearity(self, nonlin_style="lin_osc"):
        self.nonlin_style = nonlin_style

    def set_forcing(self, force):
        self.force = force

    def set_phys_params(self, params, par_type):
        self.param_type = par_type
        match par_type:
            case "constant":
                match self.nonlin_style:
                    case "lin_osc":
                        self.k = params['k']
                        self.c = params['c']
                        self.phys_params = torch.tensor([self.c, self.k])
                    case "cubic_stiffness":
                        self.k = params['k']
                        self.k3 = params['k3']
                        self.c = params['c']
                        self.phys_params = torch.tensor([self.c, self.k, self.k3])
            case "variable":
                match self.nonlin_style:
                    case "lin_osc":
                        self.register_parameter("phys_params", nn.Parameter(torch.tensor([params["c"], params["k"]])))
                    case "cubic_stiffness":
                        self.register_parameter("phys_params", nn.Parameter(torch.tensor([params["c"], params["k"], params["k3"]])))

    def set_norm_params(self, alphas, ode_norm_Lambda):
        self.alpha_t = alphas["t"]
        self.alpha_x = alphas["x"]
        self.alpha_F = alphas["F"]

        match self.nonlin_style:
            case "lin_osc":
                self.alpha_k = alphas["k"]
                self.alpha_c = alphas["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * ode_norm_Lambda,
                    "d1" : alpha_d1 * ode_norm_Lambda,
                    "d2" : alpha_d2 * ode_norm_Lambda,
                    "ff" : alpha_ff * ode_norm_Lambda
                }
                
            case "cubic_stiffness":
                self.alpha_k = alphas["k"]
                self.alpha_k3 = alphas["k3"]
                self.alpha_c = alphas["c"]

                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * ode_norm_Lambda,
                    "d0_3" : alpha_d0_3 * ode_norm_Lambda,
                    "d1" : alpha_d1 * ode_norm_Lambda,
                    "d2" : alpha_d2 * ode_norm_Lambda,
                    "ff" : alpha_ff * ode_norm_Lambda
                }
        
    def forward(self, x):
        x = self.net(x)
        return x

    def calc_residuals(self, t_pde_hat, t_obs, x_obs):

        # observation residual
        xh_obs = self.forward(t_obs)  # N_y-hat or N_y (in Ω_a)
        R_obs = xh_obs - x_obs

        # ode residual
        xh_pde_hat = self.forward(t_pde_hat)   # N_y-hat (in Ω_ode)
        dx = torch.autograd.grad(xh_pde_hat, t_pde_hat, torch.ones_like(xh_pde_hat), create_graph=True)[0]  # ∂_t-hat N_y-hat
        dx2 = torch.autograd.grad(dx, t_pde_hat, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_y-hat

        match self.param_type:
            case "constant":
                self.m_hat = self.ode_alphas["d2"]
                self.c_hat = self.ode_alphas["d1"] * self.c
                self.k_hat = self.ode_alphas["d0"] * self.k
                self.eta = self.ode_alphas["ff"]
                if self.nonlin_style == "cubic_stiffness":
                    self.k3_hat = self.ode_alphas["d0_3"] * self.k3

            case "variable":
                self.m_hat = self.ode_alphas["d2"]
                self.c_hat = self.ode_alphas["d1"] * self.phys_params[0] * self.alpha_c
                self.k_hat = self.ode_alphas["d0"] * self.phys_params[1] * self.alpha_k
                self.eta = self.ode_alphas["ff"]
                if self.nonlin_style == "cubic_stiffness":
                    self.k3_hat = self.ode_alphas["d0_3"] * self.phys_params[2] * self.alpha_k3

        match self.nonlin_style:
            case "lin_osc":
                R_pde = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat - self.eta * self.force
            case "cubic_stiffness":
                R_pde = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat + self.k3_hat * xh_pde_hat**3 - self.eta * self.force

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde
        }

    def loss_func(self, t_pde, t_obs, x_obs, lambdas):
        residuals = self.calc_residuals(t_pde, t_obs, x_obs)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_ode = lambdas['ode'] * torch.mean(R_pde**2)
        loss = L_obs + L_ode

        return loss, [L_obs, L_ode]

    def predict(self, tp):
        yp = self.forward(tp)
        return yp

class ParamClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'phys_params'):
            params = module.phys_params.data
            params = params.clamp(0,1)
            module.phys_params.data = params

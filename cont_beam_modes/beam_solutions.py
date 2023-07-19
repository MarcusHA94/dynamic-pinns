import numpy as np
from math import pi
import torch
import scipy.integrate as integrate
import toybox as tb

"""
simply supported - simply supported : ss-ss
fixed - fixed : fx-fx
free - free : fr-fr
fixed - simply : fx-s
fixed - free : fx-fr
"""

class cont_beam:

    # def __init__(self, E, I, rho, area, length):
    def __init__(self, def_type, **kwargs):
        super().__init__()

        self.mat_var_type = def_type
        self.L = self.import_variable(kwargs["l"])

        match def_type:
            case "sep_vars":
                self.E = self.import_variable(kwargs["E"])
                self.rho = self.import_variable(kwargs["rho"])
                self.I = self.import_variable(kwargs["I"])
                if type(kwargs["area"]) == list or type(kwargs["area"]) == tuple:
                    self.b = self.import_variable(kwargs["area"][0])
                    self.h = self.import_variable(kwargs["area"][1])
                    self. A = torch.product(self.import_variable(kwargs["area"]))
                    if torch.abs(self.I - (1/12)*self.b*self.h**3)/self.I < 0.01:
                        raise ValueError("Moment of inertia does not match values of b and h...")
                else:
                    self.A = self.import_variable(kwargs["area"])
                self.c = kwargs["c"]
            case "cmb_vars":
                self.EI = self.import_variable(kwargs["EI"])
                self.pA = self.import_variable(kwargs["pA"])
                self.c = self.import_variable(kwargs["c"])
    
    def import_variable(self, var):
        if torch.is_tensor(var):
            return var
        else:
            return torch.tensor(var)
            
    def gen_modes(self, bc_type, n_modes, nx):

        self.bc_type = bc_type
        self.nx = nx
        x = torch.linspace(0, self.L, nx)
        self.xx = x
        self.n_modes = n_modes
        nn = torch.arange(1, n_modes+1, 1)
        match self.mat_var_type:
            case "sep_vars":
                wn_mult = (self.E * self.I / (self.rho * self.A * self.L**4))**(0.5)
            case "cmb_vars":
                wn_mult = (self.EI / (self.pA * self.L**4))**(0.5)


        match bc_type:
            case "ss-ss":
                Cn = torch.sqrt((2/(self.pA*self.L)))
                self.bc_type_long = "simply supported - simply supported"
                beta_l = nn*pi
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = torch.zeros((self.nx, n_modes))
                self.phi_dx2_n = torch.zeros((self.nx, n_modes))
                self.phi_dx4_n = torch.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    beta_n = beta_l[n]/self.L
                    self.phi_n[:,n] = Cn * torch.sin(beta_n*x)
                    self.phi_dx2_n[:,n] = -Cn * (beta_n**2)*torch.sin(beta_n*x)
                    self.phi_dx4_n[:,n] = Cn * (beta_n**4)*torch.sin(beta_n*x)
            case "fx-fx":
                self.bc_type_long = "fixed - fixed"
                beta_l = (2*nn + 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = torch.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (torch.cos(beta_n[n]*x) - torch.cosh(beta_n[n]*x)) - \
                                    (torch.cos(beta_l[n]) - torch.cosh(beta_l[n]))/(torch.sin(beta_l[n]) - torch.sinh(beta_l[n])) * \
                                    (torch.sin(beta_n[n]*x) - torch.sinh(beta_n[n]*x))
            case "fr-fr":
                self.bc_type_long = "free - free"
                beta_l = (2*nn + 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = torch.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (torch.cos(beta_n[n]*x) + torch.cosh(beta_n[n]*x)) - \
                                    (torch.cos(beta_l[n]) - torch.cosh(beta_l[n]))/(torch.sin(beta_l[n]) - torch.sinh(beta_l[n])) * \
                                    (torch.sin(beta_n[n]*x) + torch.sinh(beta_n[n]*x))
            case "fx-ss":
                self.bc_type_long = "fixed - simply supported"
                beta_l = (4*nn + 1) * pi / 4
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
            case "fx-fr":
                self.bc_type_long = "fixed - free"
                beta_l = (2*nn - 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = torch.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (torch.cos(beta_n[n]*x) - torch.cosh(beta_n[n]*x)) - \
                                    (torch.cos(beta_l[n]) + torch.cosh(beta_l[n]))/(torch.sin(beta_l[n]) + torch.sinh(beta_l[n])) * \
                                    (torch.sin(beta_n[n]*x) - torch.sinh(beta_n[n]*x))
                    
        M = torch.zeros((self.n_modes,self.n_modes))
        K = torch.zeros((self.n_modes,self.n_modes))
        for i in range(self.n_modes):
            for j in range(self.n_modes):
                m_integrand = self.phi_n[:,i].view(-1,1) * self.phi_n[:,j].view(-1,1)
                M[i,j] = integrate.simpson(m_integrand.view(-1),self.xx)
                k_integrand = self.phi_dx2_n[:,i].view(-1,1) * self.phi_dx2_n[:,j].view(-1,1)
                K[i,j] = integrate.simpson(k_integrand.view(-1),self.xx)
        self.M = M
        self.C = self.M * self.c
        self.K = K

    def free_vibration(self, time, w0, wd0):
        nt = time.shape[0]
        x = self.xx
        nx = x.shape[0]

        ww = torch.zeros((nx, nt, self.n_modes))
        wwd = torch.zeros((nx, nt, self.n_modes))
        wwdd = torch.zeros((nx, nt, self.n_modes))
        tau = torch.zeros((2*self.n_modes,nt))
        tau_dot = torch.zeros((2*self.n_modes,nt))
        for n in range(self.n_modes):
            eta_integrand = self.pA * self.phi_n[:,n] * w0
            eta = integrate.simpson(eta_integrand, x)
            eta_dot_integrand = self.pA * self.phi_n[:,n] * wd0
            eta_dot = integrate.simpson(eta_dot_integrand, x)
            for t in range(nt):
                ww[:,t,n] = self.phi_n[:,n] * (eta*torch.cos(self.wn[n]*time[t]) + (eta_dot/self.wn[n])*torch.sin(self.wn[n]*time[t]))
                wwd[:,t,n] = self.phi_n[:,n] * (-self.wn[n]*eta*torch.sin(self.wn[n]*time[t]) + eta_dot*torch.cos(self.wn[n]*time[t]))
                wwdd[:,t,n] = self.phi_n[:,n] * (-(self.wn[n]**2)*eta*torch.cos(self.wn[n]*time[t]) - self.wn[n]*eta_dot*torch.sin(self.wn[n]*time[t]))
                tau[n,t] = eta*torch.cos(self.wn[n]*time[t]) + (eta_dot)/self.wn[n] * torch.sin(self.wn[n]*time[t])
                tau[n+self.n_modes,t] = (-self.wn[n]*eta*torch.sin(self.wn[n]*time[t]) + eta_dot*torch.cos(self.wn[n]*time[t]))
                tau_dot[n,t] = (-self.wn[n]*eta*torch.sin(self.wn[n]*time[t]) + eta_dot*torch.cos(self.wn[n]*time[t]))
                tau_dot[n+self.n_modes,t] = (-(self.wn[n]**2)*eta*torch.cos(self.wn[n]*time[t]) - self.wn[n]*eta_dot*torch.sin(self.wn[n]*time[t]))
        self.ww = ww
        self.wxt = torch.sum(ww, dim=2)
        self.wxtd = torch.sum(wwd, dim=2)
        self.wxtdd = torch.sum(wwdd, dim=2)
        self.tau = tau

        return self.wxt, self.wxtd, self.wxtdd, ww, tau, tau_dot
    
    def generate_force(self, forcing, time):

        match forcing['type']:
            case 'point_step':
                xx_id = torch.argmin(torch.abs(forcing['load_coord'] - self.xx))
                self.fxt = torch.zeros((self.xx.shape[0], time.shape[0]))
                self.fxt[xx_id,:] = forcing['force_mag'] * torch.ones(time.shape[0])

            case 'point_harmonic':
                xx_id = torch.argmin(torch.abs(forcing['load_coord'] - self.xx))
                self.fxt = torch.zeros((self.xx.shape[0], time.shape[0]))
                wr = forcing['frequency']
                self.fxt[xx_id,:] = forcing['force_mag'] * torch.sin(wr*time)

            case 'point_rpms':
                xx_id = torch.argmin(torch.abs(forcing['load_coord'] - self.xx))
                self.fxt = torch.zeros((self.xx.shape[0], time.shape[0]))
                freqs = forcing['freqs'].reshape(-1,1)
                Sx = forcing['Sx'].reshape(-1,1)
                match forcing:
                    case {'seed' : int() as seed}:
                        np.random.seed(seed)
                    case _:
                        np.random.seed(43810)
                phases = np.random.rand(freqs.shape[0], 1).reshape(-1,1)
                F_mat = np.sin(time.reshape(-1,1) @ freqs.T + phases.T)
                self.fxt[xx_id,:] = (F_mat @ Sx).view(1,-1)

            case 'point_wgn':
                xx_id = torch.argmin(torch.abs(forcing['load_coord'] - self.xx))
                self.fxt = torch.zeros((self.xx.shape[0], time.shape[0]))
                u = forcing["offset"]
                sig = forcing['force_mag']
                match forcing:
                    case {"seed" : int() as seed}:
                        np.random.seed(seed)
                    case _:
                        np.random.seed(43810)
                self.fxt[xx_id,:] = np.random.normal(u, sig, size=(time.shape[0],1))
    
    def forced_vibration(self, time, forcing):

        self.generate_force(forcing, time)

        nt = time.shape[0]
        x = self.xx
        nx = self.xx.shape[0]
        ww = torch.zeros((nx, nt, self.n_modes))
        fxt = self.fxt  # [nx, nt]
        Qnt = torch.zeros((nt, self.n_modes))
        qnt = torch.zeros((nt, self.n_modes))
        for n in range(self.n_modes):
            for tau in range(nt):
                Qt_integrand = self.phi_n[:,n] * fxt[:,tau]
                Qnt[tau,n] = integrate.simpson(Qt_integrand, x)
            for t in range(nt):
                qt_integrand = Qnt[:,n] * torch.sin(self.wn[n]*(time[t]-time))
                qnt[t,n] = 1/(self.wn[n]) * integrate.simpson(qt_integrand, time)
                ww[:,t,n] = self.phi_n[:,n] * qnt[t,n]

        return torch.sum(ww, dim=2), ww

    def init_cond_load(self, init_load):
        match init_load["type"]:
            case "point_load":
                a = init_load["load_coord"]
                b = self.L - a
                F = init_load["f0"]
                x = self.xx
                fl_id = torch.argmin(torch.abs(x - a))
                w0 = torch.zeros(x.shape[0])
                w0[:fl_id] = -((F*b*x[:fl_id])/(6*self.L*self.EI)) * (self.L**2 - b**2 - x[:fl_id]**2)
                w0[fl_id:] = -((F*b)/(6*self.L*self.EI)) * (self.L/b*(x[fl_id:]-a)**3 + (self.L**2 - b**2)*x[fl_id:] - x[fl_id:]**3)
        self.w0 = w0
        return w0


def beam_ss_simulate(time, beam, config):

    nx = beam.xx.shape[0]
    n_modes = config['n_modes']
    dt = time[1] - time[0]
    num_samps = time.shape[0]

    if config['init_state'] is None:
        tau0 = torch.zeros((2*n_modes))
    else:
        w0 = config['init_state']['w0']
        wdot0 = config['init_state']['wdot0']
        tau0 = torch.zeros((2*n_modes))
        for n in range(n_modes):
            eta_integrand = beam.pA * beam.phi_n[:,n] * w0
            eta = integrate.simpson(eta_integrand, beam.xx)
            eta_dot_integrand = beam.pA * beam.phi_n[:,n] * wdot0
            eta_dot = integrate.simpson(eta_dot_integrand, beam.xx)
            tau0[n] = eta
            tau0[n+n_modes] = eta_dot
    
    M = beam.M
    C = beam.C
    K = beam.K

    if config['forcing'] is not None:
        f = config['forcing']['F'].T  #[n_x, n_s]
    else:
        f = None

    A = torch.cat((
        torch.cat((torch.zeros((n_modes,n_modes)),torch.eye(n_modes)),axis=1),
        torch.cat((-torch.linalg.inv(M)@K, -torch.linalg.inv(M)@C),axis=1)
    ),axis=0)

    if f is not None:
        H = torch.cat((torch.zeros((n_modes,n_modes)),torch.linalg.inv(M)),axis=0)

    def rung_f(tau):
        if f is None:
            return A@tau
        else:
            return A@tau + H@f
        
    tau = torch.zeros((2*n_modes,num_samps))
    tau[:,0] = tau0

    ww = torch.zeros((nx, num_samps, n_modes))
    wwd = torch.zeros((nx, num_samps, n_modes))

    for n in range(n_modes):
        ww[:,0,n] = beam.phi_n[:,n] * tau[n,0]
        wwd[:,0,n] = beam.phi_n[:,n] * tau[n+n_modes,0]

    for t in range(num_samps-1):
        k1 = rung_f(tau[:,t])
        k2 = rung_f(tau[:,t] + k1*dt/2)
        k3 = rung_f(tau[:,t] + k2*dt/2)
        k4 = rung_f(tau[:,t] + k3*dt)
        tau[:,t+1] = tau[:,t] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
        for n in range(n_modes):
            ww[:,t+1,n] = beam.phi_n[:,n] * tau[n,t+1]
            wwd[:,t+1,n] = beam.phi_n[:,n] * tau[n+n_modes,t+1]
        
    wxt = torch.sum(ww, dim=2)
    wxtd = torch.sum(wwd, dim=2)

    return wxt, wxtd




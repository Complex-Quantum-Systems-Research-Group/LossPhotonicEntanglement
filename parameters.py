# parameters.py

class SimulationParameters:
    def __init__(self,
                 n_max=2,
                 N_spins=6,
                 omega1=1.0,
                 omega2=1.0,
                 g1=2.0,
                 g2=2.0,
                 J=-1.0,
                 delta=0.5,
                 tau_1=0.05,
                 tau_2=0.05,
                 final_evolution_time=2.0,
                 temperature_list=None,
                 delta_t_list=None):
        self.n_max = n_max
        self.N_spins = N_spins
        self.omega1 = omega1
        self.omega2 = omega2
        self.g1 = g1
        self.g2 = g2
        self.J = J
        self.delta = delta
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.final_evolution_time = final_evolution_time
        self.temperature_list = temperature_list or [0.05 + 0.05*i for i in range(40)]
        self.delta_t_list = delta_t_list or [0.5 + 0.05*i for i in range(31)]
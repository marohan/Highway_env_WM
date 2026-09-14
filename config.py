from dataclasses import dataclass

@dataclass
class ViabilityCfg:
    T: int = 8                     # horizon (s)
    dt: float = 1.0
    dx: float = 2.0                # m
    dv: float = 2.0                # m/s
    x_range: tuple = (-40.0, 280.0)
    beta: float = 0.5              # pain weight
    K: int = 5                     # world ensemble size
    pessimism_mean_weight: float = 0.15
    tie_eps: float = 0.05          # nats
    legacy_L2: bool = False
    t_forget: float = 3.0          # object permanence (s)
    unknown_mode: str = "free"     # free | penalized | blocked
    sigma_floor: float = 0.3       # m/s^2

    # DEATH symmetry (spec section 5): rear RSS inside the alive predicate.
    # False reproduces the pre-fix behaviour (gap_behind > 1.0 constant).
    rear_rss: bool = True
    v_traffic_nominal: float = 20.0
    t_react: float = 0.6
    a_other_max: float = 5.0       # assumed max |decel| of other vehicles

    # ablations
    no_option_value: bool = False  # T=1 (myopic)
    no_pessimism: bool = False     # K=1, posterior mean only
    no_pain_gradient: bool = False # beta=0
    no_memory: bool = False        # t_forget=0
    no_self_model: bool = False    # nominal constants instead of posterior
    no_sweep: bool = False         # disable lane-change swept volume

    def __post_init__(self):
        if self.no_pain_gradient:
            self.beta = 0.0
        if self.no_option_value:
            self.T = 1
        if self.no_pessimism:
            self.K = 1
        if self.no_memory:
            self.t_forget = 0.0

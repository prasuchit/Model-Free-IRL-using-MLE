class mdp:
    # Attributes of an mdp
    def __init__(self):
        self.name = None
        self.nStates = None
        self.nActions = None
        self.nFeatures = None
        self.discount = None
        self.start = None
        self.transition = None
        self.F = None
        self.weight = None
        self.reward = None
        self.piL = None
        self.VL = None
        self.QL = None
        self.dQ = None
        self.H = None
        self.alpha = None
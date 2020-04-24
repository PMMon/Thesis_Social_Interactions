class config_optimizer:
    def __init__(self, lr=1e-2):
        self.lr = lr

class config_Adam(config_optimizer):
    def __init__(self, lr=1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0):
        super(config_Adam, self).__init__(lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

class config_RMSprop(config_optimizer):
    def __init__(self, lr=1e-2, alpha = 0.99, eps=1e-8, weight_decay=0.0):
        super(config_RMSprop, self).__init__(lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay

class config_SGD(config_optimizer):
    def __init__(self, lr=1e-2, momentum=0, weight_decay=0.0, dampening=0.0, nesterov=False):
        super(config_SGD, self).__init__(lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
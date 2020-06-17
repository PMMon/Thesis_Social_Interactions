# ========== Description ===============
# Configurations for various optimizer
# ======================================

class config_optimizer:
    def __init__(self, lr=1e-2):
        self.lr = lr

class config_Adam(config_optimizer):
    """
    Defines configurations for Adam. For conceptional details of adam see:
    D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. 2014. Link: https://arxiv.org/abs/1412.6980
    """
    def __init__(self, lr=1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0):
        super(config_Adam, self).__init__(lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

class config_RMSprop(config_optimizer):
    """
    Defines configurations for RMSprop optimizer. RMSprop is an unpublished, adaptive learning rate method
    proposed by Geoff Hinton in Lecture 6e of his Coursera Class:
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self, lr=1e-2, alpha = 0.99, eps=1e-8, weight_decay=0.0):
        super(config_RMSprop, self).__init__(lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay

class config_SGD(config_optimizer):
    """
    Defines configurations for Stochastic Gradient Descent optimizer. For conceptional details of SGD, see e.g.:
    https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
    """
    def __init__(self, lr=1e-2, momentum=0, weight_decay=0.0, dampening=0.0, nesterov=False):
        super(config_SGD, self).__init__(lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
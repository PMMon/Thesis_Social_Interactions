import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base class for different Trajectory Prediction Models. Implements basic functionality, such as switching to
    different modes or saving the Model.
    """
    def __init__(self,device = 'cpu',float_type=torch.float64):
        super(BaseModel, self).__init__()

        self.__dict__.update(locals())
        self.losses = ["G_Loss"]

        self.type(float_type)
        self.gen()
        self.get_name()


    def test(self):
        """
        Change mode to testing/validation
        """
        self.eval()
        self.mode = "test"
        return self


    def get_name(self):
        """
        Get name of Model
        """
        self.model = self.__class__.__name__


    def gen(self):
        """
        Change  mode to training
        """
        self.mode = "gen"
        self.train()
        return self

    def save(self, opt, loss, epochs, path):
        """
        Save Model, the configurations of the chosen optimizer, the history of losses and the last epoch
        """
        print("Saving model to path %s ..." %path)
        torch.save({"model_state_dict": self.state_dict(), "optimizer_state_dict": opt.state_dict(), "loss": loss, "epochs": epochs}, path)





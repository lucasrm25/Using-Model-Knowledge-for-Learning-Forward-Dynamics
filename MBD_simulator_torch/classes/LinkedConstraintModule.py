import torch
from .RigidBody import RigidBody

class LinkedConstraintModule(torch.nn.Module): # ():
    def __init__(self, predBody:RigidBody, sucBody:RigidBody):
        super().__init__()
        self.setPredBody(predBody)
        self.setSucBody(sucBody)

    def getPredBody(self):
        return self._predBody[0]

    def setPredBody(self, predBody):
        self._predBody = [predBody]

    def getSucBody(self):
        return self._sucBody[0]

    def setSucBody(self, sucBody):
        self._sucBody = [sucBody]
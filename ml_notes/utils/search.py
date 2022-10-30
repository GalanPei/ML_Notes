import numpy as np

class Search(object):
    def __init__(self, prob_list) -> None:
        self.prob_list = prob_list
        
    def search(self):
        pass
        
class GreedySearch(Search):
    def __init__(self, prob_list) -> None:
        super().__init__(prob_list)
        
    def search(self):
        pass
    
class BeamSearch(Search):
    def __init__(self, prob_list, size=2) -> None:
        super().__init__(prob_list)
        self.size = size
        
    def search(self):
        pass
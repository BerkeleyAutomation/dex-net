# Abstact static methods
# Source: https://stackoverflow.com/questions/4474395/staticmethod-and-abc-abstractmethod-will-it-blend

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

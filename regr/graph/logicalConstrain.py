class LogicalConstrain:
    def __init__(self, *e):
        self.e = e
        
        if e:
            contexts = self.getContext(e[0])
            if contexts:
                context = contexts[-1]
            
                lcName = "LC%i"%(len(context.logicalConstrains))
                context.logicalConstrains[lcName] = self
    def getContext(self, e):
        if isinstance(e, LogicalConstrain):
            return self.getContext(e.e[0])
        else:
            return e._context
            
class andL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)

class orL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
class ifL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
class existsL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
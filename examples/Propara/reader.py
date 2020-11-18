import os
from regr.data.reader import RegrReader


class ProparaReader(RegrReader):
    def getprocedureIDval(self, item):
        return item['id']

    def getBodyval(self, item):
        return item['body']

    def getstepsval(self, item):
        return item['steps_rel'], item['numbers'], item['raw']

    def getentityval(self, item):
        return item['entity']

    def getentity_stepval(self, item):
        return item['entity_step']

    def getnon_existenceval(self, item):
        return item['non_existence']

    def getknown_locationval(self, item):
        return item['known_location']

    def getunknown_locationval(self, item):
        return item['unknown_location']
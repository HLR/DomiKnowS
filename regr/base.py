from collections import defaultdict
import pprint


class AutoNamed(object):
    _name_counter = defaultdict(list)

    @classmethod
    def clear(cls):
        cls._name_counter.clear()

    @classmethod
    def get(cls, name, value=None):
        lst = cls._name_counter.get(name)
        if lst:
            return lst[0]  # the fist has exact the name
        else:
            return value

    def assign_suggest_name(self, name):
        cls = self.__class__
        if name is None:
            name = cls.__name__
        if name in cls._name_counter:
            while True:
                name_attempt = '{}-{}'.format(name,
                                              len(cls._name_counter[name]))
                if name_attempt in cls._name_counter:
                    cls._name_counter[name].append(
                        cls._name_counter[name_attempt])
                else:
                    break
            name = name_attempt

        assert name not in cls._name_counter
        self.name = name
        # each name may have two reference...
        cls._name_counter[name].append(self)

    def __init__(self, name=None):
        self.assign_suggest_name(name)

    _repr_level = 0

    @property
    def _repr_scope(self):
        return self.__class__

    def __repr__(self):
        cls = self.__class__
        self._repr_scope._repr_level += 1  # in case there is recursive invoke
        if self._repr_scope._repr_level > 1 or not callable(getattr(self, 'what', None)):

            repr_str = '{class_name}(name=\'{name}\')'.format(class_name=cls.__name__,
                                                              name=self.name)
        else:
            repr_str = '{class_name}(name=\'{name}\', what={what})'.format(class_name=cls.__name__,
                                                                           name=self.name,
                                                                           what=pprint.pformat(self.what(), width=8))
        self._repr_scope._repr_level -= 1
        return repr_str

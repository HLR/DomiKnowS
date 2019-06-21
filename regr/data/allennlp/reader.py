from typing import Iterator
from collections import OrderedDict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from ...utils import optional_arg_decorator, optional_arg_decorator_for

field_dict = {}  # just a global container

FIELDS_SUFFIX = '()'


def get_field_decorator(name):
    def update(key):
        if key.endswith(FIELDS_SUFFIX):
            raise ValueError(
                'Data field name cannot end with "{}"'.format(FIELDS_SUFFIX))

        def up(func):
            def update_field(self_, fields, *args, **kwargs):
                try:
                    field = func(self_, fields, *args, **kwargs)
                    if field is not None:
                        fields[key] = field
                except KeyError:
                    pass
                finally:
                    return fields
            field_dict[name][key] = update_field
            return update_field
        return up
    return update


def get_fields_decorator(name):
    def update_each(func):
        def update_field(self_, fields, *args, **kwargs):
            for key, field in func(self_, fields, *args, **kwargs):
                try:
                    if field is not None:
                        fields[key] = field
                except KeyError:
                    pass
            return fields
        field_dict[name][func.__name__ + FIELDS_SUFFIX] = update_field
        return update_field
    return update_each


class SensableReaderMeta(type):
    # Steps constructing class:
    # 0. Resolving MRO entries
    # 1. Determining the appropriate metaclass
    # 2. Preparing the class namespace

    def __prepare__(name, bases, **kwds):
        namespace = OrderedDict()
        namespace['field'] = get_field_decorator(name)
        namespace['fields'] = get_fields_decorator(name)
        field_dict[name] = OrderedDict()
        return namespace

    # 3. Executing the class body
    # 4. Creating the class object
    # def __call__(name, bases, namespace, **kwds)
    #   def __new__(name, bases, namespace, **kwds) -> cls
    #     4.1.0 colloct descriptors
    #     4.1.1 for each descriptor call __set_name__(self, owner, name)
    #     4.1.2 init cls
    def __init__(cls, name, bases, namespace):
        super(SensableReaderMeta, cls).__init__(name, bases, namespace)

    # 5. Nothing more, ready to go


class SensableReader(DatasetReader, metaclass=SensableReaderMeta):
    __metaclass__ = SensableReaderMeta

    def _to_instance(self, raw_sample) -> Instance:
        cls = type(self)
        fields = {}

        for key, update_field in field_dict[cls.__name__].items():
            update_field(self, fields, raw_sample)
        return Instance(fields)

    def _read(self, *args, **kwargs) -> Iterator[Instance]:
        for raw_sample in self.raw_read(*args, **kwargs):
            yield self._to_instance(raw_sample)


@optional_arg_decorator_for(lambda cls: issubclass(cls, SensableReader))
def keep_fields(cls, *keys):
    fields = OrderedDict()
    # though same level, reversed order seems more expected
    for base in reversed(cls.__bases__):
        if issubclass(base, SensableReader):
            if len(keys) == 0:
                fields.update(field_dict[base.__name__])
            else:
                for key in keys:
                    if key in field_dict[base.__name__]:
                        fields[key] = field_dict[base.__name__][key]
    for key in keys:
        if key not in fields:
            raise ValueError(
                'Cannot find key {} from any of the bases of {}'.format(key, cls.__name__))
    # update with current lastly
    fields.update(field_dict[cls.__name__])
    field_dict[cls.__name__] = fields

    return cls

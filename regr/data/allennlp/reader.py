from typing import Iterator
from collections import OrderedDict, defaultdict
import uuid
from six import with_metaclass
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
import cls
from ...utils import optional_arg_decorator, optional_arg_decorator_for


PLURAL_SUFFIX = '()'


class SensableReaderMeta(cls.ClsMeta):
    def __init__(cls, name, bases, namespace):
        cls.tokens_dict = OrderedDict()
        cls.textfield_dict = OrderedDict()
        cls.field_dict = OrderedDict()
        super(SensableReaderMeta, cls).__init__(name, bases, namespace)


class SensableReader(with_metaclass(SensableReaderMeta, DatasetReader)):
    def __init__(self, lazy=False) -> None:
        super(SensableReader, self).__init__(lazy=lazy)
        self.key_sensors = defaultdict(list)
        self.key_tokens = {}
        self.token_indexers = defaultdict(dict)

    def _to_instance(self, raw_sample) -> Instance:
        cls = type(self)
        fields = {}

        # prepare tokens
        for key, update_tokens in cls.tokens_dict.items():
            update_tokens(self, fields, raw_sample)

        # prepare `TextField`s
        for key, update_textfield in cls.textfield_dict.items():
            update_textfield(self, fields, raw_sample)

        # prepare other field
        for key, update_field in cls.field_dict.items():
            update_field(self, fields, raw_sample)

        return Instance(fields)

    def _read(self, *args, **kwargs) -> Iterator[Instance]:
        for raw_sample in self.raw_read(*args, **kwargs):
            yield self._to_instance(raw_sample)

    @cls
    def field(cls, key):
        if key.endswith(PLURAL_SUFFIX):
            raise ValueError(
                'Data field key cannot end with "{}"'.format(PLURAL_SUFFIX))

        def decorator(func):
            def update_field(self_, fields, raw_sample):
                field = func(self_, fields, raw_sample)
                if field is not None:
                    for sensor in self_.key_sensors[key]:
                        fields[sensor.fullname] = field
                return fields
            cls.field_dict[key] = update_field
            return update_field
        return decorator

    @cls
    def tokens(cls, key):
        if key.endswith(PLURAL_SUFFIX):
            raise ValueError(
                'Data field key cannot end with "{}"'.format(PLURAL_SUFFIX))

        def decorator(func):
            def update_tokens(self_, fields, raw_sample):
                tokens = func(self_, fields, raw_sample)
                if tokens is not None:
                    self_.key_tokens[key] = tokens
                return tokens
            cls.tokens_dict[key] = update_tokens
            return update_tokens
        return decorator

    @cls
    def textfield(cls, key):
        if key.endswith(PLURAL_SUFFIX):
            raise ValueError(
                'Data field name cannot end with "{}"'.format(PLURAL_SUFFIX))

        def decorator(func):
            def update_textfield(self_, fields, raw_sample):
                for sensor in self_.key_sensors[key]: # sensors associate with 'pos_tag'
                    # it would be possible different sensor refers to different tokens
                    tokens = self_.key_tokens[sensor.tokens_key]
                    field = func(self_, fields, tokens)
                    if field is not None:
                        fields[sensor.fullname] = field
                return fields
            cls.textfield_dict[key] = update_textfield
            return update_textfield
        return decorator

    @cls(True)
    def fields(cls):
        def decorator(func):
            def update_field(self_, fields, raw_sample):
                for key, field in func(self_, fields, raw_sample):
                    if field is not None:
                        for sensor in self_.key_sensors[key]:
                            fields[sensor.fullname] = field
                return fields
            cls.field_dict[func.__name__ + PLURAL_SUFFIX] = update_field
            return update_field
        return decorator

    def claim(self, key, sensor):
        self.key_sensors[key].append(sensor)

    def get_fieldname(self, key):
        return self.key_sensors[key][0].fullname # using any[0]?


@optional_arg_decorator_for(lambda cls: issubclass(cls, SensableReader))
def keep_fields(cls, *keys):
    tokens_dict = OrderedDict()
    textfield_dict = OrderedDict()
    field_dict = OrderedDict()

    # reversed order ensure MRO priority by override
    for base in reversed(cls.__bases__):
        if issubclass(base, SensableReader):
            if len(keys) == 0:
                tokens_dict.update(base.tokens_dict)
                textfield_dict.update(base.textfield_dict)
                field_dict.update(base.field_dict)
            else:
                for key in keys:
                    if key in base.tokens_dict:
                        tokens_dict[key] = base.tokens_dict[key]
                    if key in base.textfield_dict:
                        textfield_dict[key] = base.textfield_dict[key]
                    if key in base.field_dict:
                        field_dict[key] = base.field_dict[key]
    for key in keys:
        if (key not in tokens_dict) and (key not in textfield_dict) and (key not in field_dict):
            raise ValueError('Cannot find key {} from any of the bases of {}'.format(key, cls.__name__))

    # update with current lastly
    tokens_dict.update(cls.tokens_dict)
    textfield_dict.update(cls.textfield_dict)
    field_dict.update(cls.field_dict)

    cls.tokens_dict = tokens_dict
    cls.textfield_dict = textfield_dict
    cls.field_dict = field_dict

    return cls

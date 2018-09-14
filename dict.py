# Inspired by allennlp implementation at https://github.com/allenai/allennlp/blob/master/allennlp/common/params.py
def flatten(nested_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the parameters of a flat dictionary from keys to values.
    Nested structure is collapsed with periods.
    """
    flat_dict = {}

    def recurse(parameters, path):
        for key, value in parameters.items():
            newpath = path + [key]
            if isinstance(value, dict):
                recurse(value, newpath)
            else:
                flat_dict['.'.join(newpath)] = value

    recurse(nested_dict, [])
    return flat_dict


# Inspired by allennlp implementation at https://github.com/allenai/allennlp/blob/master/allennlp/common/params.py
def unflatten(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a "flattened" dict with compound keys, e.g.
        {"a.b": 0}
    unflatten it:
        {"a": {"b": 0}}
    """
    unflat: Dict[str, Any] = {}

    for compound_key, value in flat_dict.items():
        curr_dict = unflat
        parts = compound_key.split(".")
        for key in parts[:-1]:
            curr_value = curr_dict.get(key)
            if key not in curr_dict:
                curr_dict[key] = {}
                curr_dict = curr_dict[key]
            elif isinstance(curr_value, dict):
                curr_dict = curr_value
            else:
                raise ConfigurationError("flattened dictionary is invalid")
        if not isinstance(curr_dict, dict) or parts[-1] in curr_dict:
            raise ConfigurationError("flattened dictionary is invalid")
        else:
            curr_dict[parts[-1]] = value

    return unflat


def nested_dict_iter_1(nested):
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(value)
        else:
            yield key, value


def nested_dict_iter_2(nested):
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(top_level, value)
        elif isinstance(value, list):
            for value_item in value:
                yield key, value

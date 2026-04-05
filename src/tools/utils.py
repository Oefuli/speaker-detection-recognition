

def remap_dict_keys(
        obj_in: dict | list[dict]
        ) -> dict | list[dict]:
    """
    (1) Maps a dictionary to a list of dictionaries.
    Application e.g. if the keys of the dictionaries are made of tuples.

    (2) Maps a list of dictionaries to one dictionary. Reverse of (1).

    Args:
        obj_in (dict or list): See above!
    """

    # (1)
    if isinstance(obj_in, dict):
        return [{'key':k, 'value': v} for k, v in obj_in.items()]

    elif isinstance(obj_in, list):

        new_dict = {}

        for key_val in obj_in:

            new_dict[tuple(key_val['key'])] = key_val['value']

        return new_dict

# ---------------------------------------------------------- #
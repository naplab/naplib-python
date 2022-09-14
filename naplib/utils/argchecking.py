from itertools import groupby

from ..data import Data

def _parse_outstruct_args(data,
                          *args,
                          allow_different_lengths=False,
                          allow_strings_without_outstruct=True):
    '''
    Parse a variable number of arguments and return a tuple with either the field
    from the Data which is specified by a given arg or the arg itself if it
    does not correspond to a field in the Data.
    
    Parameters
    ----------
    data : Data or other Object
        If a naplib.Data object, then will check if the other args are fields
        of this object, otherwise they will be returned without change.
    *args : variable number of inputs
        Other arguments. Each one will be checked to see if it is a field of the
        Data, and if it is then it will be replaced by that field. If it is a
        list, it will be returned without change, and if it is any other type, it will
        be repeated into a list of the same length as the Data or any other list args,
        and if nothing else is provided, it will be return as is.
    allow_different_lengths : bool, default=False
        If True, args are not required to all have the same length, and they do not need
        to be the same length as the Data if one is provided.
    allow_strings_without_outstruct : bool | list of bool (length len(args)), default=True
        If True, then string args that are passed in are accepted and returned back
        even if no Data was provided. 

    Returns
    -------
    *args_out : single arg or tuple of args, same length as *args input
    '''
    # if no Data was provided, then just parse the args on their own
    if not isinstance(data, Data):
        args = list(args)
        if isinstance(allow_strings_without_outstruct, bool):
            allow_strings_without_outstruct = [allow_strings_without_outstruct for _ in range(len(args))]
        if len(args) > 1:
            lengths = []
            for i, arg in enumerate(args):
                if isinstance(arg, list):
                    lengths.append(len(arg))
                if isinstance(arg, str) and not allow_strings_without_outstruct[i]:
                    raise ValueError(f'Input is a string but no Data object was provided from which to select a field based on the string.')
            if len(lengths) > 0:
                if not allow_different_lengths and not _all_equal_list(lengths):
                    raise ValueError(f"The supplied list arguments have different lengths, but they must all be the same length.")
                for i, arg in enumerate(args):
                    if not isinstance(arg, list) and not isinstance(arg, tuple):
                        args[i] = [arg for _ in range(lengths[0])]
            return tuple(args)
        else:
            return args[0]
    
    # if Data was provided, check if args are fields of the Data
    args_out = []
    
    data_fieldnames = data.fields
    for arg in args:
        if isinstance(arg, str):
            if arg in data_fieldnames:
                args_out.append(data.get_field(arg))
            else:
                raise ValueError(f"{arg} is a string but is not a field of the Data.")
        else:
            if isinstance(arg, list):
                if not allow_different_lengths and len(arg) != len(data):
                    raise ValueError(f"The list argument supplied is length {len(arg)}, which is not"
                                     " the same as the provided Data, which is length {len(Data)}.")
                args_out.append(arg)
            elif isinstance(arg, tuple):
                args_out.append(arg)
            else:
                args_out.append([arg for _ in range(len(data))])
    if len(args_out) > 1:
        return tuple(args_out)
    return args_out[0]
                            
def _all_equal_list(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


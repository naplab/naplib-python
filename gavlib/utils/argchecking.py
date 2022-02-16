from ..out_struct import OutStruct

def _parse_outstruct_args(outstruct, *args):
    '''
    Parse a variable number of arguments and return a tuple with either the field
    from the outstruct which is specified by a given arg or the arg itself if it
    does not correspond to a field in the outstruct.
    
    Parameters
    ----------
    outstruct : any type or None
        If a naplib.OutStruct object, then will check if the other args are fields
        of this object, otherwise they will be returned without change.
    args : variable number of inputs
        Other arguments. Each one will be checked to see if it is a field of the
        outstruct, and if it is then it will be replaced by that field, otherwise
        it will be returned without change.

    Returns
    -------
    out : single arg or tuple of args, same length as *args input
    '''
    if not isinstance(outstruct, OutStruct):
        if len(args) > 1:
            return args
        else:
            return args[0]
    
    args_out = []
    
    outstruct_fieldnames = outstruct.fields
    for arg in args:
        if arg in outstruct_fieldnames:
            args_out.append(outstruct.get_field(arg))
        else:
            args_out.append(arg)
    if len(args_out) > 1:
        return tuple(args_out)
    return args_out[0]


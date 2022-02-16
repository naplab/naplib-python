from ..out_struct import OutStruct

def _parse_outstruct_args(outstruct, *args):
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


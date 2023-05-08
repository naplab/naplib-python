from collections.abc import Iterable, Sequence
import numpy as np
from mne import Info

STRICT_FIELDS_REQUIRED = set(['name','sound','soundf','resp','dataf'])

class OutStruct(Iterable):
    '''
    Class for storing electrode response data along with
    task- and electrode-related variables. Under the hood, it consists
    of a list of dictionaries where each dictionary contains all the data
    for one trial.

    Please see the :ref:`example notebooks <working with outstructs>` for more detailed
    tutorials which demonstrate using the OutStruct object in different types of analysis.
    
    Parameters
    ----------
    data : dict or list of dictionaries
        The Nth ictionary defines the Nth trial data, typically for the Nth stimulus.
        Each dictionary must contain the same keys if passed in a list of multiple trials.
    strict : bool, default=False
        If True, requires strict adherance to the following standards:
        1) Each trial must contain at least the following fields:
        ['name','sound','soundf','resp','dataf']
        2) Each trial must contain the exact same set of fields
    
    Attributes
    ----------
    fields : list of strings
        Field names in the data.
    mne_info : mne.Info instance
        Measurement info object containing things like electrode locations
        (only if OutStruct is created from reading a file format like BIDS).
    info : dict
        Extra info (not trial-specific) that a user wants to store
        using outstruct.set_info or outstruct.update_info

    
    '''
    def __init__(self, data, strict=False):
        
        if isinstance(data, dict):
            data = [data]
            self._data = data
        elif isinstance(data, list):
            self._data = data
        else:
            raise TypeError(f'Can only create OutStruct from a dict or a list '
                            f'of dicts, but found type {type(data)}')
        self._strict = strict
        self._validate_new_out_data(data, strict=strict)
        self._info = dict()
        self._mne_info = None

                
    def set_field(self, fielddata, fieldname):
        '''
        Parameters
        ----------
        fielddata : list
            List containing data to add to each trial for this field. Must 
            be same length as this object
        fieldname : string
            Name of field to add. If this field already exists in the OutStruct
            then the current field will be overwritten.
        '''
        if not isinstance(fielddata, list):
            raise TypeError(f'Input data must be a list, but found {type(fielddata)}')
        if len(fielddata) != len(self):
            raise Exception('Length of field is not equal to length of this OutStruct')
        for i in range(len(self.data)):
            self.data[i][fieldname] = fielddata[i]
            
    def get_field(self, fieldname):
        '''
        Return all trials for a single field.
        
        Parameters
        ----------
        fieldname : string
            Which field to get.
        Returns
        -------
        field : list
            List containing each trial's value for this field.
        '''
        try:
            return [tmp[fieldname] for tmp in self.data]
        except KeyError:
            raise KeyError(f'Invalid fieldname: {fieldname} not found in data.')
            
    def __getitem__(self, index):
        '''
        Get either a trial or a field using bracket indexing. See examples
        below for details.

        Parameters
        ----------
        index : int or string
            Which trial to get, or which field to get.

        Returns
        -------
        data : dict, list, or OutStruct
            If index is an integer, returns the corresponding trial as a dict. If index
            is a string, returns the corresponding field, and if it is a list of strings,
            returns those fields together in a new OutStruct object.

        Examples
        --------
        >>> # Get a specific trial based on its index, which returns a dict
        >>> from naplib import OutStruct
        >>> trial_data = [{'name': 'Zero', 'trial': 0, 'resp': [[0,1],[2,3]]},
        ...               {'name': 'One', 'trial': 1, 'resp': [[4,5],[6,7]]}]
        >>> out = OutStruct(trial_data, strict=False)
        >>> out[0]
        {'name': 'Zero', 'trial': 0, 'resp': [[0, 1], [2, 3]]}

        >>> # Get a slice of trials, which returns an OutStruct object
        >>> out[:2]
        OutStruct of 2 trials containing 3 fields
        [{"name": <class 'str'>, "trial": <class 'int'>, "resp": <class 'list'>}
        {"name": <class 'str'>, "trial": <class 'int'>, "resp": <class 'list'>}]

        >>> # Get a list of trial data from a single field
        >>> out['name']
        ['TrialZero', 'TrialOne']
        >>> out[0]
        {'name': 'TrialZero', 'trial': 0, 'resp': [[0, 1], [2, 3]]}

        >>> # Get multiple fields using a list of fieldnames, which returns an OutStruct containing that subset of fields
        >>> out[['resp','trial']]
        OutStruct of 2 trials containing 2 fields
        [{"resp": <class 'list'>, "trial": <class 'int'>}
        {"resp": <class 'list'>, "trial": <class 'int'>}]
        '''
        if isinstance(index, slice):
            return OutStruct(self.data[index], strict=self._strict)
        if isinstance(index, str):
            return self.get_field(index)
        if isinstance(index, list) or isinstance(index, np.ndarray):
            if isinstance(index[0], str):
                return OutStruct([dict([(field, x[field]) for field in index]) for x in self], strict=False)
            else:
                return OutStruct([self.data[i] for i in index], strict=False)
        try:
            # TODO: change this to return a type OutStruct if you do slicing - problem with trying to
            # print because it says KeyError for self.data[0] for key 0
            return self.data[index]
        except IndexError:
            raise IndexError(f'Index invalid for this data. Tried to index {index} but length is {len(self)}.')
        
            
    def __setitem__(self, index, data):
        '''
        Set a specific trial or set of trials, or set a specific field, using
        bracket indexing. See examples below for details.

        Parameters
        ----------
        index : int or string
            Which trial to set, or which field to set. If an integer, must be <= the
            length of the OutStruct, since you can only set a currently existing trial
            or append to the end, but you cannot set a trial that is beyond that. 
        data : dict or list of data
            Either trial data to add or field data to add. If index is an
            integer, dictionary should contain all the same fields as
            current OutStruct object.

        Examples
        --------
        >>> # Set a field of an OutStruct
        >>> from naplib import OutStruct
        >>> trial_data = [{'name': 'Zero', 'trial': 0, 'resp': [[0,1],[2,3]]},
        ...               {'name': 'One', 'trial': 1, 'resp': [[4,5],[6,7]]}]
        >>> out = OutStruct(trial_data)
        >>> out[0] = {'name': 'New', 'trial': 10, 'resp': [[0,-1],[-2,-3]]}
        >>> out[0]r 
        {'name': 'New', 'trial': 10, 'resp': [[0, -1], [-2, -3]]}

        >>> # We can also set all values of a field across trials
        >>> out['name'] = ['TrialZero','TrialOne']
        >>> out['name']
        ['TrialZero', 'TrialOne']
        '''
        if isinstance(index, str):
            self.set_field(data, index)
        else:
            if index > len(self):
                raise IndexError((f'Index is too large. Current data is length {len(self)} '
                    'but tried to set index {index}. If you want to add to the end of the list '
                    'of trials, use the OutStruct.append() method.'))
            elif index == len(self):
                self.append(data)
            else:
                self.data[index] = data
     
    def append(self, trial_data, strict=None):
        '''
        Append a single trial of data to the end of an OutStruct.
        
        Parameters
        ----------
        trial_data : dict
            Dictionary containing all the same fields as current OutStruct object.
        strict : bool, default=self._strict
            If true, enforces that new data contains the exact same set of fields as
            the current OutStruct. Default value is self._strict, which is set based
            on the input when creating a new OutStruct from scratch with __init__()

        Raises
        ------
        TypeError
            If input data is not a dict.
        ValueError
            If strict is `True` and the fields contained in the trial_data do
            not match the fields currently stored in the OutStruct.

        Examples
        --------
        >>> # Set a field of an OutStruct
        >>> from naplib import OutStruct
        >>> trial_data = [{'name': 'Zero', 'trial': 0, 'resp': [[0,1],[2,3]]},
        ...               {'name': 'One', 'trial': 1, 'resp': [[4,5],[6,7]]}]
        >>> out = OutStruct(trial_data)
        >>> new_trial_data = {'name': 'Two', 'trial': 2, 'resp': [[8,9],[10,11]]}
        >>> out.append(new_trial_data)
        >>> len(out)
        3
        '''
        if strict is None:
            strict = self._strict
        self._validate_new_out_data([trial_data], strict=strict)
        self.data.append(trial_data)
        
    def set_info(self, info):
        '''
        Set the info dict for this OutStruct. If there is already data in the
        `info` attribute, it is replaced with this.
        
        Parameters
        ----------
        info : dict
            Dictionary containing info to store in the OutStruct's `info` attribute.
            
        '''
        if not isinstance(info, dict):
            raise TypeError(f'info must be a dict but got {type(info)}')
        self._info = info
        
    def update_info(self, info):
        '''
        Add data from a dict to this object's `info` attribute. If there is already data in the
        `info` attribute, this new info is simply added. Keys which exist in the current
        `info` dict and also in this new dict will be replaced, while others will be kept.
        
        Parameters
        ----------
        info : dict
            Dictionary containing info to add to the OutStruct's `info` attribute.
        '''
        self._info.update(info)
        
    def set_mne_info(self, info):
        '''
        Set the mne_info attribute, which contains measurement information.
        
        Parameters
        ----------
        info : mne.Info instance
            Info to set.
        '''
        if not isinstance(info, Info):
            raise TypeError(f'input info must be an instance of mne.Info, but got {type(info)}')
        self._mne_info = info
    
    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        '''
        Get the number of trials in the OutStruct with ``len(outstruct)``.

        Examples
        --------
        >>> from naplib import OutStruct
        >>> trial_data = [{'trial': 0, 'resp': [[0,1],[2,3]]},
                          {'trial': 1, 'resp': [[4,5],[6,7]]}]
        >>> out = OutStruct(trial_data, strict=False)
        >>> len(out)
        2
        '''
        return len(self.data)
    
    def __repr__(self):
        return self.__str__() # until we can think of a better __repr__
    
    def __str__(self):
        to_return = f'OutStruct of {len(self)} trials containing {len(self.fields)} fields\n['
        
        to_print = 2 if len(self) > 3 else 3
        for trial_idx, trial in enumerate(self[:to_print]):
            fieldnames = list(trial.keys())
            to_return += '{'
            for f, fieldname in enumerate(fieldnames):
#                 to_return += f'"{fieldname}": {trial[fieldname].__str__()}'
                to_return += f'"{fieldname}": {type(trial[fieldname])}'
                if f < len(fieldnames)-1:
                    to_return += ', '
            if trial_idx < len(self)-1:
                to_return += '}\n'
            else:
                to_return += '}'
        if to_print == 3:
             to_return += ']\n'
        elif to_print == 2:
            to_return += '\n...\n{'
            fieldnames = list(self[-1].keys())
            for f, fieldname in enumerate(fieldnames):
#                 to_return += f'"{fieldname}": {self[-1][fieldname].__str__()}'
                to_return += f'"{fieldname}": {type(self[-1][fieldname])}'
                if f < len(fieldnames)-1:
                    to_return += ', '
            to_return += '}]'
        return to_return
    
    def _validate_new_out_data(self, input_data, strict=True):
        
        first_trial_fields = set(self.fields)
        for t, trial in enumerate(input_data):
            if not isinstance(trial, dict):
                raise TypeError(f'input data is not a list of dicts, found {type(trial)}')
            trial_fields = set(trial.keys())
            if strict and trial_fields != first_trial_fields:
                raise ValueError(f'New data does not contain the same fields as the first trial.')        
            if strict:
                for required_field in STRICT_FIELDS_REQUIRED:
                    if required_field not in trial_fields:
                        raise ValueError(f'For a "strict" OutStruct, the data does not contain the required field {required_field}.')
    
    @property
    def fields(self):
        '''List of strings containing names of all fields in this OutStruct.'''
        return [k for k, _ in self._data[0].items()]
    
    @property
    def data(self):
        '''List of dictionaries containing data for each stimulus
        response and all associated variables.'''
        return self._data

    @property
    def info(self):
        '''Dictionary which can be used to store metadata info which does not
        change over trials, such as subject, recording, or task information.'''
        return self._info
    
    @property
    def mne_info(self):
        '''
        `mne.Info <https://mne.tools/dev/generated/mne.Info.html>`_ instance
        which stores measurement information and can be used with mne's visualization
        functions. This is empty by default unless it is manually added or read in
        by a function like `naplib.io.read_bids`.
        '''
        if self._mne_info is None:
            raise ValueError('No mne_info is available for this OutStruct. This must '
                             'be read in from external data or added manually to the OutStruct.')
        return self._mne_info
    
def join_fields(outstructs, fieldname='resp', axis=-1, return_outstruct=False):
    '''
    Join trials from a field of multiple OutStruct objects by zipping them
    together and concatenating each trial together. The field must be of type
    np.ndarray and concatenation is done with np.concatenate().
    
    Parameters
    ----------
    outstructs : sequence of OutStructs
        Sequence containing the different outstructs to join
    fieldname : string, default='resp'
        Name of the field to concatenate from each OutStruct. For each trial in
        each outstruct, this field must be of type np.ndarray or something which
        can be input to np.concatenate().
    axis : int, default = -1
        Axis along which to concatenate each trial's data. The default corresponds
        to the channel dimension of the conventional 'resp' field of an OutStruct.
    return_outstruct : bool, default=False
        If True, returns data as an OutStruct with a single field named fieldname.

    Returns
    -------
    joined_data : list of np.ndarrays, or OutStruct
        Joined data of same length as each of the outstructs containing concatenated data
        for each trial.
    '''
    
    for out in outstructs:
        if not isinstance(out, OutStruct):
            raise TypeError(f'All inputs must be an OutStruct but found {type(out)}')
        field = out.get_field(fieldname)
        if not isinstance(field[0], np.ndarray):
            raise TypeError(f'Can only concatenate np.ndarrays, but found {type(field[0])} in this field')

    starting_fields = [out.get_field(fieldname) for out in outstructs] # each one should be a list of np.arrays
    
    to_return = []
    
    zipped_fields = list(zip(*starting_fields))
    for i, field_set in enumerate(zipped_fields):
        to_return.append(np.concatenate(field_set, axis=axis))
        
    if return_outstruct:
        return OutStruct([dict([(fieldname, x)]) for x in to_return], strict=False)
    return to_return
        
    

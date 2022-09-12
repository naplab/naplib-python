from collections.abc import Iterable, Sequence
from itertools import groupby
import numpy as np
from mne import Info


STRICT_FIELDS_REQUIRED = set(['name','sound','soundf','resp','dataf'])


class Data(Iterable):
    '''
    Class for storing electrode response data along with
    task- and electrode-related variables. Under the hood, it consists
    of a list of dictionaries where each dictionary contains all the data
    for one trial.

    Please see the :ref:`example notebooks <working with data>` for more detailed
    tutorials which demonstrate using the Data object in different types of analysis.
    
    Parameters
    ----------
    data : dict or list of dictionaries
        If a list of dicts, then the Nth dictionary defines the Nth trial data, typically
        corresponding to the Nth stimulus. Each dictionary must contain the same keys if
        passed in as a list of multiple trials. If a single dict, then the keys specify the
        field names and the values specify the data across trials, and each value must be
        a list of length num_trials.
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
        (only if Data is created from reading a file format like BIDS).
    info : dict
        Extra info (not trial-specific) that a user wants to store
        using Data.set_info or Data.update_info

    Notes
    -----
    .. figure:: /figures/naplib-python-data-figure.png
        :width: 500px
        :alt: Data object layout
        :align: center

    The above is a depiction of the type of data that might be stored in an
    instance of the Data class. Any number of trials can be stored with any
    number and type of fields. Responses and information do not need to be
    aligned or the same length/shape across trials. Information can be retrieved
    from the Data instance by trial, by field, or by a combination of the two,
    using bracket indexing and slicing, as described below.
    
    '''
    def __init__(self, data, strict=False):
        
        if isinstance(data, dict):
            lengths = []
            for k, v in data.items():
                if not isinstance(v, list):
                    raise TypeError(f'When creating an Data from a dict, each value in the '
                                     'dict must be a list, but for key "{k}" got type {type(v)}')
                lengths.append(len(v))
            if not _all_equal_list(lengths):
                raise ValueError(f'When creating an Data from a dict, each value in the '
                                  'dict must be a list of the same length, but got different lengths: {lengths}')
            data = [dict(zip(data, vals)) for vals in zip(*data.values())]
            self._data = data
        elif isinstance(data, list):
            self._data = data
        else:
            raise TypeError(f'Can only create Data from a dict or a list '
                            f'of dicts, but found type {type(data)}')
        self._strict = strict
        self._validate_new_out_data(data, strict=strict)
        self._info = dict()
        self._mne_info = None

                
    def set_field(self, fielddata, fieldname):
        '''
        Set the information in a single field with a new list of data.

        Parameters
        ----------
        fielddata : list
            List containing data to add to each trial for this field. Must 
            be same length as this object
        fieldname : string
            Name of field to add. If this field already exists in the Data
            then the current field will be overwritten.
        '''
        if not isinstance(fielddata, list):
            raise TypeError(f'Input data must be a list, but found {type(fielddata)}')
        if len(fielddata) != len(self):
            raise Exception('Length of field is not equal to length of this Data')
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
        data : dict, list, or Data
            If index is an integer, returns the corresponding trial as a dict. If index
            is a string, returns the corresponding field, and if it is a list of strings,
            returns those fields together in a new Data object.

        Examples
        --------
        >>> # Get a specific trial based on its index, which returns a dict
        >>> from naplib import Data
        >>> trial_data = [{'name': 'Zero', 'trial': 0, 'resp': [[0,1],[2,3]]},
        ...               {'name': 'One', 'trial': 1, 'resp': [[4,5],[6,7]]}]
        >>> data = Data(trial_data, strict=False)
        >>> data[0]
        {'name': 'Zero', 'trial': 0, 'resp': [[0, 1], [2, 3]]}

        >>> # Get a slice of trials, which returns an Data object
        >>> out[:2]
        Data object of 2 trials containing 3 fields
        [{"name": <class 'str'>, "trial": <class 'int'>, "resp": <class 'list'>}
        {"name": <class 'str'>, "trial": <class 'int'>, "resp": <class 'list'>}]

        >>> # Get a list of trial data from a single field
        >>> data['name']
        ['TrialZero', 'TrialOne']
        >>> data[0]
        {'name': 'TrialZero', 'trial': 0, 'resp': [[0, 1], [2, 3]]}

        >>> # Get multiple fields using a list of fieldnames, which returns an Data containing that subset of fields
        >>> data[['resp','trial']]
        Data object of 2 trials containing 2 fields
        [{"resp": <class 'list'>, "trial": <class 'int'>}
        {"resp": <class 'list'>, "trial": <class 'int'>}]
        '''
        if isinstance(index, slice):
            return Data(self.data[index], strict=self._strict)
        if isinstance(index, str):
            return self.get_field(index)
        if isinstance(index, list) or isinstance(index, np.ndarray):
            if isinstance(index[0], str):
                return Data([dict([(field, x[field]) for field in index]) for x in self], strict=False)
            else:
                return Data([self.data[i] for i in index], strict=False)
        try:
            # TODO: change this to return a type Data if you do slicing - problem with trying to
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
            length of the Data, since you can only set a currently existing trial
            or append to the end, but you cannot set a trial that is beyond that. 
        data : dict or list of data
            Either trial data to add or field data to add. If index is an
            integer, dictionary should contain all the same fields as
            current Data object.

        Examples
        --------
        >>> # Set a field of an Data
        >>> from naplib import Data
        >>> trial_data = [{'name': 'Zero', 'trial': 0, 'resp': [[0,1],[2,3]]},
        ...               {'name': 'One', 'trial': 1, 'resp': [[4,5],[6,7]]}]
        >>> data = Data(trial_data)
        >>> data[0] = {'name': 'New', 'trial': 10, 'resp': [[0,-1],[-2,-3]]}
        >>> data[0]
        {'name': 'New', 'trial': 10, 'resp': [[0, -1], [-2, -3]]}

        >>> # We can also set all values of a field across trials
        >>> data['name'] = ['TrialZero','TrialOne']
        >>> data['name']
        ['TrialZero', 'TrialOne']
        '''
        if isinstance(index, str):
            self.set_field(data, index)
        else:
            if index > len(self):
                raise IndexError((f'Index is too large. Current data is length {len(self)} '
                    'but tried to set index {index}. If you want to add to the end of the list '
                    'of trials, use the Data.append() method.'))
            elif index == len(self):
                self.append(data)
            else:
                self.data[index] = data
     
    def append(self, trial_data, strict=None):
        '''
        Append a single trial of data to the end of an Data.
        
        Parameters
        ----------
        trial_data : dict
            Dictionary containing all the same fields as current Data object.
        strict : bool, default=self._strict
            If true, enforces that new data contains the exact same set of fields as
            the current Data. Default value is self._strict, which is set based
            on the input when creating a new Data from scratch with __init__()

        Raises
        ------
        TypeError
            If input data is not a dict.
        ValueError
            If strict is `True` and the fields contained in the trial_data do
            not match the fields currently stored in the Data.

        Examples
        --------
        >>> # Set a field of an Data
        >>> from naplib import Data
        >>> trial_data = [{'name': 'Zero', 'trial': 0, 'resp': [[0,1],[2,3]]},
        ...               {'name': 'One', 'trial': 1, 'resp': [[4,5],[6,7]]}]
        >>> data = Data(trial_data)
        >>> new_trial_data = {'name': 'Two', 'trial': 2, 'resp': [[8,9],[10,11]]}
        >>> data.append(new_trial_data)
        >>> len(data)
        3
        '''
        if strict is None:
            strict = self._strict
        self._validate_new_out_data([trial_data], strict=strict)
        self.data.append(trial_data)
        
    def set_info(self, info):
        '''
        Set the info dict for this Data. If there is already data in the
        `info` attribute, it is replaced with this.
        
        Parameters
        ----------
        info : dict
            Dictionary containing info to store in the Data's `info` attribute.
            
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
            Dictionary containing info to add to the Data's `info` attribute.
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
        Get the number of trials in the Data object with ``len(Data)``.

        Examples
        --------
        >>> from naplib import Data
        >>> trial_data = [{'trial': 0, 'resp': [[0,1],[2,3]]},
                          {'trial': 1, 'resp': [[4,5],[6,7]]}]
        >>> data = Data(trial_data, strict=False)
        >>> len(data)
        2
        '''
        return len(self.data)
    
    def __repr__(self):
        return self.__str__() # until we can think of a better __repr__
    
    def __str__(self):
        to_return = f'Data object of {len(self)} trials containing {len(self.fields)} fields\n['
        
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
            to_return += '}]\n'
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
                        raise ValueError(f'For a "strict" Data object, the data does not contain the required field {required_field}.')
    
    @property
    def fields(self):
        '''List of strings containing names of all fields in this Data.'''
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
            raise ValueError('No mne_info is available for this Data. This must '
                             'be read in from external data or added manually to the Data.')
        return self._mne_info
    
def join_fields(data_list, fieldname='resp', axis=-1, return_as_data=False):
    '''
    Join trials from a field in multiple Data objects by zipping them
    together and concatenating each trial together. The field must be of type
    np.ndarray and concatenation is done with np.concatenate().
    
    Parameters
    ----------
    data : sequence of Data instances
        Sequence containing the different Data objects to join.
    fieldname : string, default='resp'
        Name of the field to concatenate from each Data object. For each trial in
        each Data instance, this field must be of type np.ndarray or something which
        can be input to np.concatenate().
    axis : int, default = -1
        Axis along which to concatenate each trial's data. The default corresponds
        to the channel dimension of the conventional 'resp' field of a Data object.
    return_as_data : bool, default=False
        If True, returns data as a Data object with a single field named fieldname.

    Returns
    -------
    joined_data : list of np.ndarrays, or Data instance
        Joined data of same length as each of the Data objects containing concatenated data
        for each trial.
    '''
    
    for out in data_list:
        if not isinstance(out, Data):
            raise TypeError(f'All inputs to data_list must be Data instance but found {type(out)}')
        field = out.get_field(fieldname)
        if not isinstance(field[0], np.ndarray):
            raise TypeError(f'Can only concatenate np.ndarrays, but found {type(field[0])} in this field')

    starting_fields = [out.get_field(fieldname) for out in data_list] # each one should be a list of np.arrays
    
    to_return = []
    
    zipped_fields = list(zip(*starting_fields))
    for i, field_set in enumerate(zipped_fields):
        to_return.append(np.concatenate(field_set, axis=axis))
        
    if return_as_data:
        return Data([dict([(fieldname, x)]) for x in to_return], strict=False)
    return to_return
        
def _all_equal_list(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

from collections.abc import Iterable
import numpy as np

class OutStruct(Iterable):
    '''
    Class for storing electrode response data along with
    task- and electrode-related variables. Under the hood, it consists
    of a list of dictionaries where each dictionary contains all the data
    for one trial.
    
    Parameters
    ----------
    data : list of dictionaries
        The Nth ictionary defines the trial_data for the Nth stimulus.
        Each dictionary must contain the same keys.
    strict : bool, default=True
        If True, requires strict adherance to the following standard
            - Each trial must contain at least the following fields:
              ['name','sound','soundf','resp','dataf']
            - Each trial must contain the exact same set of fields
        
    Methods
    -------
    set_field(field, fieldname)
    get_field(fieldname)
    append(trial_data)
    
    Attributes
    ----------
    fields : list of strings
        Names of all fields stored in this OutStruct
    data : list of dictionaries
        Data for each stimulus response and all associated variables
    
    '''
    def __init__(self, data, strict=True):
        
        self._data = data
        self._strict = strict
        self._validate_new_out_data(data, strict=strict)

                
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
        Returns
        -------
        '''
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
        Parameters
        ----------
        index : int
            Which trial to get.
        Returns
        -------
        trial_data : dict
            Returns the index'th 
        '''
        if isinstance(index, str):
            return self.get_field(index)
        if isinstance(index, list):
            return OutStruct([dict([(field, x[field]) for field in index]) for x in self], strict=False)
        try:
            return self.data[index]
        except IndexError:
            raise IndexError(f'Index invalid for this data.')
        

            
    def __setitem__(self, index, trial_data):
        '''
        Parameters
        ----------
        index : int
            Which trial to get.
        trial_data : dict
            Dictionary containing all the same fields as current OutStruct object.
        Returns
        -------
        '''
        if isinstance(index, str):
            self.set_field(trial_data, index)
        else:
            if index >= len(self):
                raise IndexError((f'Index is too large. Current data is length {len(self)} '
                    'but tried to set index {index}. If you want to add to the end of the list '
                    'of trials, use the OutStruct.append() method.'))
            else:
                self.data[index] = trial_data
     
    def append(self, trial_data, strict=None):
        '''
        Append trial data to end of OutStruct.
        
        Parameters
        ----------
        trial_data : dict
            Dictionary containing all the same fields as current OutStruct object.
        strict : bool, default=self._strict
            If true, enforces that new data contains the exact same set of fields as
            the current OutStruct. Default value is self._strict, which is set based
            on the input when creating a new OutStruct from scratch with __init__()

        Returns
        -------
        '''
        if strict is None:
            strict = self._strict
        self._validate_new_out_data([trial_data], strict=strict)
        self.data.append(trial_data)
        
    def __iter__(self):
        return (self[i] for i in range(len(self)))

#     def __next__(self):
#         if self._iter_n < len(self):
#             self._iter_n += 1
#             return (self[i] for i in range(len(self)))
#         else:
#             raise StopIteration

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return self.__str__() # until we can think of a better __repr__
    
    def __str__(self):
        to_return = f'OutStruct of {len(self)} trials containing {len(self.fields)} fields\n['
        fieldnames = self.fields
        to_print = 2 if len(self) > 3 else 3
        for trial in self[:to_print]:
            to_return += '{'
            for f, fieldname in enumerate(fieldnames):
#                 to_return += f'"{fieldname}": {trial[fieldname].__str__()}'
                to_return += f'"{fieldname}": {type(trial[fieldname])}'
                if f < len(fieldnames)-1:
                    to_return += ', '
            to_return += '}\n'
        if to_print == 2:
            to_return += '...\n'
            for f, fieldname in enumerate(fieldnames):
#                 to_return += f'"{fieldname}": {self[-1][fieldname].__str__()}'
                to_return += f'"{fieldname}": {type(self[-1][fieldname])}'
                if f < len(fieldnames)-1:
                    to_return += ', '
            to_return += '}'
        return to_return
    
    def _validate_new_out_data(self, input_data, strict=True):
        
        first_trial_fields = set(self.fields)
        for t, trial in enumerate(input_data):
            if not isinstance(trial, dict):
                raise TypeError(f'input data is not a list of dicts, found {type(trial)}')
            trial_fields = set(trial.keys())
            if strict and trial_fields != first_trial_fields:
                raise ValueError(f'New data does not contain the same fields as the first trial.')        
        
    @property
    def fields(self):
        '''Get names of all fields in this object'''
        return [k for k, _ in self.data[0].items()]
    
    @property
    def data(self):
        return self._data

    
def concatenate_outstructs(outstructs=None, fieldname='resp', axis=-1):
    '''
    Concatenate trials from a field of multiple OutStruct objects by zipping them
    together and concatenating each trial together.
    
    Parameters
    ----------
    outstructs
    '''
    
    
    
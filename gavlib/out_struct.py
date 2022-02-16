import numpy as np

class OutStruct(object):
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
    
    Returns
    -------
    self : returns an instance of self
    
    '''
    def __init__(self, data):
        
        self._data = data
                
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
        if index >= len(self):
            raise IndexError((f'Index is too large. Current data is length {len(self)} '
                'but tried to set index {index}. If you want to add to the end of the list '
                'of trials, use the OutStruct.append() method.'))
        else:
            self.data[index] = trial_data
     
    def append(self, trial_data):
        '''
        Append trial data to end of OutStruct.
        
        Parameters
        ----------
        trial_data : dict
            Dictionary containing all the same fields as current OutStruct object.

        Returns
        -------
        '''
        self.data.append(trial_data)
        
    def __iter__(self):
        self._iter_n = 0
        return self

    def __next__(self):
        if self._iter_n < len(self):
            self._iter_n += 1
            return self[self._iter_n-1]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.data)
    
    @property
    def fields(self):
        '''Get names of all fields in this object'''
        return [k for k, _ in self.data[0].items()]
    
    @property
    def data(self):
        return self._data


"""
A configuration-file parser.

A configuration file consists of categories, delimited by a header [categories], 
and followed by parameters and entries designated by "parameter = value".

The output is a dictionnary with categories as keys, and values are dictionnaries that
each contain the parameters (sub-keys) and values (sub-values).


To use, first generate the class, then read the configuration file :
>>> config = Config()
>>> c = config.read(myconfigfile.conf)
>>> c
{'category1' : {'item1' : value1, 'item2' : value2', ...},
 'category2' : {'item3' : value1, ...}, 
 ...}

    Requirements
    ------------
    sys
    os


About :
configreader is class inspired by the library configparser, available for Python 3, 
adapted for our algorithm. We use the same structure, and some common attributes, 
but this is a lighter and less complete tool than the original one.
See :
https://github.com/python/cpython/blob/3.9/Lib/configparser.py

Adapted by Mathieu Le Verge--Serandour
November 2020
"""

import sys
import os



class Config() :
    """
    Config()
    
        Config class. Creates empty directories self.config and self.categories.
    """
    def __init__(self) :
        self.config = {}
        self.categories = []
    
    def read_file(self, filename) :
        """
        self.read_file(filename)
        
            Parameters
            ----------
            filename : string
                Name of the config file
            Returns
            -------
            s : string
                The content of the config file, in list form, one entry per line of the file.
        """
        f = open(filename, 'r')
        s = f.readlines()
        f.close()
        return s

    def read_config(self, string) :
        """
        self.read_config(string)
        
            Read the config file.
            
            Parameters
            ----------
            string : list
                List of string that compose the configuration file 
        
            Returns
            -------
            config : dict
                Dictionnary of the entries, grouped by categories
                Example : {'category1' : {'item1' : value1, 'item2' : value2', ...},
                           'category2' : {'item3' : value1, ...}, 
                            ...}
            cat_list : list
                List of the categories, such as ['category1', 'category2', ...]
        """
        config = {}
        cat_list = []
        
        for line in string :
            if line != '\n' :
                # Detects the categories
                #if line[0].startswith('[') and line[-2].startswith(']') :
                if line[0] == '[' and line[-2] == ']' :
                    # Detect subcategories
                    if line[0:2] == '[[' and line[-3:-1] == ']]' :
                        subcat = line[2:-3]
                        config[cat][subcat] = {}
                    else :
                        cat = line[1:-2]
                        config[cat] = {}
                        cat_list += [str(cat)]
                        subcat = None
                        
                # Detects the entries under the last category
                else :
                    if not line.startswith('#') :
                        res = line.split('=')
                        arg = res[0].replace(" ", "")
                        val = res[1].replace("\n", "").replace(" ", "")
                        if subcat is None :
                            config[cat][arg] = val
                        else :
                            config[cat][subcat][arg] = val
                    
        return config, cat_list

    def read(self, filename) :
        """
        self.read(filename)
        
            Read the configuration file, creates the categories
        
            Parameters
            ----------
            filename : string
                Name of the configuration file.
        
            Returns
            -------
            self.config : configuration entries
                The content of the configuration file, stored in a dictionnary with categories as keys.                
        """
        string_file = self.read_file(filename)
        self.config, self.categories = self.read_config(string_file)
        return self.config
    
    def write(self, filename, keys_order = []) :
        """
        self.write(filename, keys_order)
        
            Write the configuration in a file. Order of the keys and entries can be provided.
            
            Parameters
            ----------
            filename : string
                Name where to write the configuration file.
            keys_order : list, optional
                If provided, gives an order in which writing the categories, and their entries.
        """
        f = open(filename, 'w')
        if len(keys_order) == 0 :
            for key in self.config.keys() :
                f.write('['+str(key)+']\n')
                for sub_key in self.config[key].keys() :
                    f.write(str(sub_key) + ' = ' + str(self.config[key][sub_key]) + '\n')
                f.write('\n')
        else :
            
            for line in keys_order :
                key = line[0]
                sub_keys_order = line[1]
                f.write('['+str(key)+']\n')
                for sub_key in sub_keys_order :
                    f.write(str(sub_key) + ' = ' + str(self.config[key][sub_key]) + '\n')
                f.write('\n')
        f.close()
    
    def add(self, category, name, value) :
        """
        self.add(category, name, value)
            
            Add an entry to the config, given its category, name and value.
        
            Parameters
            ----------
            category : string
                Category of the entry in the configuration. If it does not already exists, 
                the category is created and added to self.categories list.
            name : string
                Name of the entry in the configuration.
            value : string
                Value of the entry. Will be written as a string.
        
        """
        if category in self.categories :
            self.config[category][name] = str(value)
        else :
            self.categories += [str(name)]
            self.config[category] = {}
            self.config[category][name] = str(value)
            
    def __str__(self) :
        """
        self.__str__()
        
            Print the category. The entries with a category are not necessarily ordered.
        """
        for key in self.config.keys() :
            print('['+str(key)+']')
            for sub_key in self.config[key].keys() :
                print(str(sub_key) + ' = ' + str(self.config[key][sub_key]) )
            print('')
        return ''
        
    def has_option(self, option1, option2='') :
        """
        self.has_option(option1, option2=')
            
            Checks whether an entry is in the configuration, returns True if so.
            If option2 is specified, option1 corresponds to the category, 
            option2 corresponds to the name.
            Otherwise, option1 is simply the category.
            
            Parameter
            ---------
            option1 : string
                Name of the option1 entry.
            option2 : string, optional, default : ''
                Name of the option2 entry.
            
        """
        if len(option2) == 0 :
            if option1 in self.categories() :
                return True
        else :
            if option1 in self.config.keys() and option2 in self.config[option1].keys() :
                return True
        return False
        
    def get_item(self, key, subkey) :
        """
        self.get_item(key, subkey)
        
            Returns the value of the item under config[key][subkey] in the configuration.
        """
        return self.config[key][subkey]
        
    def set_item(self, key, subkey, value) :
        """
        self.set_item(key, subkey, value)
            
            Sets the old value of the item under config[key][subkey] 
            in the configuration to its new value.
            
            Parameter
            ---------
            key : string
                Category of the entry
            subkey : string
                Name of the entry
            value : string
                New value for the entry
            
        """
        self.config[key][subkey] = value
        return ;

#

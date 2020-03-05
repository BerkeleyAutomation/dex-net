"""
YAML Configuration Parser
Author : Jeff Mahler
"""
import os
import ruamel.yaml as yaml
import re
from collections import OrderedDict

class YamlConfig(object):
    """Class to load a configuration file and parse it into a dictionary.

    Attributes
    ----------
    config : :obj:`dictionary`
        A dictionary that contains the contents of the configuration.
    """

    def __init__(self, filename=None):
        """Initialize a YamlConfig by loading it from the given file.

        Parameters
        ----------
        filename : :obj:`str`
            The filename of the .yaml file that contains the configuration.
        """
        self.config = {}
        if filename:
            self._load_config(filename)

    def keys(self):
        """Return the keys of the config dictionary.

        Returns
        -------
        :obj:`list` of :obj:`Object`
            A list of the keys in the config dictionary.
        """
        return self.config.keys()

    def update(self, d):
        """Update the config with a dictionary of parameters.

        Parameters
        ----------
        d : :obj:`dict`
            dictionary of parameters
        """
        self.config.update(d)

    def __contains__(self, key):
        """Overrides 'in' operator.
        """
        return key in self.config.keys()

    def __getitem__(self, key):
        """Overrides the key access operator [].
        """
        return self.config[key]

    def __setitem__(self, key, val):
        """Overrides the keyed setting operator [].
        """
        self.config[key] = val

    def iteritems(self):
        """Returns iterator over config dict.
        """
        return self.config.iteritems()

    def save(self, filename):
        """ Save a YamlConfig to disk. """
        yaml.dump(self, open(filename, 'w'))

    def _load_config(self, filename):
        """Loads a yaml configuration file from the given filename.

        Parameters
        ----------
        filename : :obj:`str`
            The filename of the .yaml file that contains the configuration.
        """
        # Read entire file for metadata
        fh = open(filename, 'r')
        self.file_contents = fh.read()

        # Replace !include directives with content
        config_dir = os.path.split(filename)[0]
        include_re = re.compile('^(.*)!include\s+(.*)$', re.MULTILINE)

        def recursive_load(matchobj, path):
            first_spacing = matchobj.group(1)
            other_spacing = first_spacing.replace('-', ' ')
            fname = os.path.join(path, matchobj.group(2))
            new_path, _ = os.path.split(fname)
            new_path = os.path.realpath(new_path)
            text = ''
            with open(fname) as f:
                text = f.read()
            text = first_spacing + text
            text = text.replace('\n', '\n{}'.format(other_spacing), text.count('\n') - 1)
            return re.sub(include_re, lambda m : recursive_load(m, new_path), text)

#        def include_repl(matchobj):
#            first_spacing = matchobj.group(1)
#            other_spacing = first_spacing.replace('-', ' ')
#            fname = os.path.join(config_dir, matchobj.group(2))
#            text = ''
#            with open(fname) as f:
#                text = f.read()
#            text = first_spacing + text
#            text = text.replace('\n', '\n{}'.format(other_spacing), text.count('\n') - 1)
#            return text

        self.file_contents = re.sub(include_re, lambda m : recursive_load(m, config_dir), self.file_contents)
        # Read in dictionary
        self.config = self.__ordered_load(self.file_contents)

        # Convert functions of other params to true expressions
        for k in self.config.keys():
            self.config[k] = YamlConfig.__convert_key(self.config[k])

        fh.close()

        # Load core configuration
        return self.config

    @staticmethod
    def __convert_key(expression):
        """Converts keys in YAML that reference other keys.
        """
        if type(expression) is str and len(expression) > 2 and expression[1] == '!':
            expression = eval(expression[2:-1])
        return expression

    def __ordered_load(self, stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
        """Load an ordered dictionary from a yaml file.

        Note
        ----
        Borrowed from John Schulman.
        http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts/21048064#21048064"
        """
        class OrderedLoader(Loader):
            pass
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            lambda loader, node: object_pairs_hook(loader.construct_pairs(node)))
        return yaml.load(stream, OrderedLoader)

    def __iter__(self):
        # Converting to a `list` will have a higher memory overhead, but realistically there
        # should not be *that* many keys.
        self._keys = list(self.config.keys())
        return self

    def __next__(self):
        try:
            return self._keys.pop(0)
        except IndexError:
            raise StopIteration

    next = __next__  # For Python 2.

"""
Class for autocomplete on prompts.
Author: Jeff Mahler
"""
import os
import re
import readline

RE_SPACE = re.compile('.*\s+$', re.M)

class Completer(object):
    """
    Tab completion class for Dex-Net CLI.
    Adapted from http://stackoverflow.com/questions/5637124/tab-completion-in-pythons-raw-input
    """
    def __init__(self, commands=[]):
        """ Provide a list of commands """
        self.commands = commands
        self.prefix = None

        # dexnet entity tab completion
        self.words = []

    def _listdir(self, root):
        "List directory 'root' appending the path separator to subdirs."
        res = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                name += os.sep
            res.append(name)
        return res

    def _complete_path(self, path=None):
        "Perform completion of filesystem path."
        if path is None or path == '':
            return self._listdir('./')
        dirname, rest = os.path.split(path)
        tmp = dirname if dirname else '.'
        res = [os.path.join(dirname, p)
                for p in self._listdir(tmp) if p.startswith(rest)]
        # more than one match, or single match which does not exist (typo)
        if len(res) > 1 or not os.path.exists(path):
            return res
        # resolved to a single directory, so return list of files below it
        if os.path.isdir(path):
            return [os.path.join(path, p) for p in self._listdir(path)]
        # exact file match terminates this completion
        return [path + ' ']

    def complete_extra(self, args):
        "Completions for the 'extra' command."
        # treat the last arg as a path and complete it
        if len(args) == 0:
            return self._listdir('./')            
        return self._complete_path(args[-1])

    def complete(self, text, state):
        "Generic readline completion entry point."

        # dexnet entity tab completion
        results = [w for w in self.words if w.startswith(text)] + [None]
        if results != [None]:
            return results[state]

        buffer = readline.get_line_buffer()
        line = readline.get_line_buffer().split()

        # dexnet entity tab completion
        results =  [w for w in self.words if w.startswith(text)] + [None]
        if results != [None]:
            return results[state]

        # account for last argument ending in a space
        if RE_SPACE.match(buffer):
            line.append('')

        return (self.complete_extra(line) + [None])[state]

    # dexnet entity tab completion
    def set_words(self, words):
        self.words = [str(w) for w in words]

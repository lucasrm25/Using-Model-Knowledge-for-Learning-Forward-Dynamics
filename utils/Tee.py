import os, sys
import traceback

class Tee(object):
    # https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "a")
    
    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        self.file.flush()
        self.terminal.flush()
        sys.stdout = self.terminal
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, message):
        self.terminal.write(message)
        # self.terminal.flush()
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        pass
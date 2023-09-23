import ast
import argparse

def arg_as_list(string):
    v = ast.literal_eval(string)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \ '%s\' is not a list" % (string))

    return v
import os

def print_tree(directory, level=0):
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        print('    ' * level + '|-- ' + entry)
        if os.path.isdir(path):
            print_tree(path, level + 1)

# Use the function with the current directory or specify a directory
print_tree('.')

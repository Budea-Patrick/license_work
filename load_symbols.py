def load_symbols(filename='symbols.meta'):
    symbols = {}
    with open(filename, 'r') as file:
        exec(file.read(), globals())
        symbols = globals()['symbols']
    return symbols

def load_symbols(filename='symbols.meta'):
    symbols = {}
    with open(filename, 'r') as file:
        content = []
        for line in file:
            line = line.strip()
            if not line.startswith('#') and line:
                content.append(line)
        exec('\n'.join(content), globals())
        symbols = globals().get('symbols', {})
    return symbols

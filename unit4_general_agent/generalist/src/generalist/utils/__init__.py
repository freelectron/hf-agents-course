import textwrap

def pprint(text):
    wrapped_lines = textwrap.wrap(text, width=130)
    for line in wrapped_lines:
        print(line)

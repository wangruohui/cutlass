
# import logging
# def info(s):
#     print(s)



# FORMAT = '%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s'

import difflib
import torch


def diff_strings(a: str, b: str, *, use_loguru_colors: bool = False) -> str:
    """
    https://gist.github.com/ines/04b47597eb9d011ade5e77a068389521
    """
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    if use_loguru_colors:
        green = '<GREEN><black>'
        red = '<RED><black>'
        endgreen = '</black></GREEN>'
        endred = '</black></RED>'
    else:
        green = '\x1b[38;5;16;48;5;2m'
        red = '\x1b[38;5;16;48;5;1m'
        yellow = "\033[93m"
        endgreen = '\x1b[0m'
        endred = '\x1b[0m'
        endyellow = '\x1b[0m'

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            output.append(f"{yellow}{a[a0:a1]}{endyellow}")
        elif opcode == 'insert':
            output.append(f'{green}{b[b0:b1]}{endgreen}')
        elif opcode == 'delete':
            output.append(f'{red}{a[a0:a1]}{endred}')
        elif opcode == 'replace':
            output.append(f'{green}{b[b0:b1]}{endgreen}')
            output.append(f'{red}{a[a0:a1]}{endred}')
    return ''.join(output)

    # for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
    #     if opcode == 'equal':
    #         output0.append(f'{yellow}{a[a0:a1]}{endyellow}')
    #         output1.append(f'{yellow}{b[b0:b1]}{endyellow}')
    #     elif opcode == 'insert':
    #         output0.append(f'{green}{a[a0:a1]}{endgreen}')
    #         output1.append(f'{green}{b[b0:b1]}{endgreen}')
    #     elif opcode == 'delete':
    #         output0.append(f'{red}{a[a0:a1]}{endred}')
    #         output1.append(f'{red}{b[b0:b1]}{endred}')
    #     elif opcode == 'replace':
    #         output0.append(f'{green}{a[a0:a1]}{endgreen}')
    #         output1.append(f'{red}{b[b0:b1]}{endred}')
    # return [''.join(output0), ''.join(output1)]


def diff_tensor(a:torch.Tensor, b:torch.Tensor) -> None:
    stra = str(a).splitlines()
    strb = str(a).splitlines()

    ret = []
    for la, lb in zip(stra, strb):
        ret.append(diff_strings(la, lb))
        # ret.append(la)

    ret = "\n".join(ret)
    print(ret)

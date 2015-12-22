import ranking


def generate_modules():

    modules = [ranking]

    for mod in modules:
        fname, module = mod.generate_module()

        print('Writing module {} to {}'.format(mod.__name__,
                                               fname))
        module.write(fname)


if __name__ == '__main__':

    generate_modules()

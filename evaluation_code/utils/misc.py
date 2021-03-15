import os
import glob


def get_filenames(source, extension):
    # If extension is a list
    if source is None:
        return []
    # Seamlessy load single file, list of files and files from directories.
    source_fns = []
    if isinstance(source, str):
        if os.path.isdir(source):
            if not isinstance(extension, str):
                for fmt in extension:
                    source_fns += get_filenames(source, fmt)
            else:
                source_fns = sorted(
                    glob.glob("{}/**/*{}".format(source, extension), recursive=True))
        elif os.path.isfile(source):
            source_fns = [source]
    elif len(source) and isinstance(source[0], str):
        for s in source:
            source_fns.extend(get_filenames(s, extension=extension))
    return source_fns


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

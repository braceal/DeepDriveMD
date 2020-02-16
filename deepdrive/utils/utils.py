def get_id(filename, prefix, ext):
    """
    Given a path to a file in the form of <path>/<prefix><id>.<ext>
    returns <id>

    Parameters
    ----------
    filename : str
        full path or basepath to `prefix` prefixed file name
        with extension `ext`

    prefix : str
        prefix of file name

    ext : str
        extension of file

    Example
    -------
    filename = './DeepDriveMD/data/val-loss-0-timestamp-98.npy'
    id_ = get_id(filename, prefix='val-loss-', ext='npy')
    print(id_) -> 0-timestamp-98

    """
    if prefix not in filename:
        raise Exception(f'prefix: {prefix} not in filename: {filename}')
    if ext not in filename:
        raise Exception(f'ext: {ext} not in filename: {filename}')

    return filename.split(prefix)[1].split(ext)[0][:-1]


def get_id(filename, ext='h5', sep='-'):
	"""
	Given a path to a file or a plain file name of the
	form <path>/part1<sep>part2...<sep>partn<sep>id.<ext>
	returns id as a int.

	Parameters
	----------
	filename : str
		full path or basepath to `sep` delimited file name 
		with extension `ext`

	ext : str
		extension of file

	sep : str
		delimiter of file name

	Example
	-------
	filename = './DeepDriveMD/data/val-loss-0.npy'
	id_ = get_id(filename, ext='npy')
	print(id_) -> 0

	"""
	return int(filename.split(f'.{ext}')[-2].split(sep)[-1])

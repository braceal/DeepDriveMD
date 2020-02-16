import pytest

from deepdrive.utils import get_id

class TestGetId:

    @classmethod
    def setup_class(self):
        pass

    def test_get_id(self):
        prefix = 'filename-'
        id1 = '1-1-4'
        ext = 'h5'
        file = f'path/{prefix}{id1}.{ext}'

        assert get_id(file, prefix, ext) == id1

        prefix = 'arbitrary-fname/with/slashes'
        id1 = 'arbitrary-string-id71920.any.string.'
        ext = 'any-extension'
        file = f'path/{prefix}{id1}.{ext}'

        assert get_id(file, prefix, ext) == id1

        try:
            get_id(file, prefix + 'invalid', ext)
            assert False
        except:
            pass
        try:
            get_id(file, prefix, 'npy')
            assert False
        except:
            pass

    @classmethod
    def teardown_class(self):
        pass

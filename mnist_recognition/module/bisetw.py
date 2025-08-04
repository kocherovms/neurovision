import biset
import numpy as np

class bisetw:
    def __init__(self):
        self.inst = biset.biset_create()

    def __del__(self):
        biset.biset_destroy(self.inst)

    def __len__(self):
        return biset.biset_get_len(self.inst)

    def __contains__(self, key):
        assert key.ndim == 1, key.ndim

        key = np.require(key, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        return biset.biset_contains(self.inst, key.shape[0], key)

    def get(self, key):
        assert key.ndim == 1, key.ndim

        key = np.require(key, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        return biset.biset_get(self.inst, key.shape[0], key)

    def add_many(self, keys, key_sizes=None):
        assert keys.ndim == 2, keys.ndim

        if not key_sizes is None:
            assert key_sizes.ndim == 1, key_sizes.ndim
            assert key_sizes.shape[0] == keys.shape[0]
        else:
            key_sizes = np.full(keys.shape[0], keys.shape[1], dtype=np.int32)
            
        keys = np.require(keys, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        key_sizes = np.require(key_sizes, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        is_added_boolmap = np.require(np.zeros(keys.shape[0], dtype=bool), requirements=['C_CONTIGUOUS', 'WRITEABLE'])
        biset.biset_add_many(self.inst, keys.shape[0], keys.shape[1], key_sizes, keys, is_added_boolmap)
        return is_added_boolmap

    def add(self, key):
        assert key.ndim == 1, key.ndim

        key = np.require(key, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        return biset.biset_add(self.inst, key.shape[0], key)

    def remove_many(self, keys, key_sizes=None):
        assert keys.ndim == 2, keys.ndim

        if not key_sizes is None:
            assert key_sizes.ndim == 1, key_sizes.ndim
            assert key_sizes.shape[0] == keys.shape[0]
        else:
            key_sizes = np.full(keys.shape[0], keys.shape[1], dtype=np.int32)

        keys = np.require(keys, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        key_sizes = np.require(key_sizes, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        is_removed_boolmap = np.require(np.zeros(keys.shape[0], dtype=bool), requirements=['C_CONTIGUOUS', 'WRITEABLE'])
        biset.biset_remove_many(self.inst, keys.shape[0], keys.shape[1], key_sizes, keys, is_removed_boolmap)
        return is_removed_boolmap

    def remove(self, key):
        assert key.ndim == 1, key.ndim

        key = np.require(key, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        biset.biset_remove(self.inst, key.shape[0], key)

    def replace_many(self, keys_from, keys_to, key_from_sizes=None, key_to_sizes=None):
        assert keys_from.ndim == 2, keys_from.ndim

        if not key_from_sizes is None:
            assert key_from_sizes.ndim == 1, key_from_sizes.ndim
            assert key_from_sizes.shape[0] == key_from_sizes.shape[0]
        else:
            key_from_sizes = np.full(keys_from.shape[0], keys_from.shape[1], dtype=np.int32)
        
        assert keys_to.ndim == 2, keys_to.ndim

        if not key_to_sizes is None:
            assert key_to_sizes.ndim == 1, key_to_sizes.ndim
            assert key_to_sizes.shape[0] == key_to_sizes.shape[0]
        else:
            key_to_sizes = np.full(keys_to.shape[0], keys_to.shape[1], dtype=np.int32)

        assert keys_from.shape[0] == keys_to.shape[0]

        keys_from = np.require(keys_from, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        key_from_sizes = np.require(key_from_sizes, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        keys_to = np.require(keys_to, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        key_to_sizes = np.require(key_to_sizes, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        is_replaced_boolmap = np.require(np.zeros(keys_from.shape[0], dtype=bool), requirements=['C_CONTIGUOUS', 'WRITEABLE'])
        biset.biset_replace_many(
            self.inst, keys_from.shape[0], 
            keys_from.shape[1], key_from_sizes, keys_from,
            keys_to.shape[1], key_to_sizes, keys_to,
            is_replaced_boolmap
        )
        return is_replaced_boolmap

    def replace(self, key_from, key_to):
        assert key_from.ndim == 1, key_from.ndim
        assert key_to.ndim == 1, key_to.ndim

        key_from = np.require(key_from, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        key_to = np.require(key_to, dtype=np.int32, requirements=['C_CONTIGUOUS'])
        
        return biset.biset_replace(self.inst, key_from.shape[0], key_from, key_to.shape[0], key_to)

    def clear(self):
        biset.biset_clear(self.inst)


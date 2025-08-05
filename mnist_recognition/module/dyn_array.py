import numpy as np
from heapq import heapify, heappush, heappop

class DynArray(object):
    def __init__(self, N, xp, initial_length=10, dtype=None, grow_policy=None, observer=None):
        self.xp = xp
        self.N = N
        self.initial_length = initial_length
        self.array = xp.zeros((self.initial_length, N), dtype=dtype)
        self.free_indices = list(range(self.array.shape[0]))
        heapify(self.free_indices)
        self.leased_indices = set()
        self.max_leased_index = -1
        self.grow_policy = (lambda current_array_size: current_array_size * 2) if grow_policy is None else grow_policy
        self.observer = observer
        self.__notify_observer()

    @property
    def len(self):
        return len(self.leased_indices)

    @property 
    def active_len(self):
        return self.max_leased_index + 1

    def lease(self):
        if self.free_indices:
            index = heappop(self.free_indices) # lowest possible index is returned, so the array space is reused effectively
            assert not index in self.leased_indices 
            self.leased_indices.add(index)
            self.max_leased_index = max(self.max_leased_index, index)
            return index

        current_array_size = self.array.shape[0]
        new_array_size = self.grow_policy(current_array_size)
        assert new_array_size > current_array_size
        
        self.array = self.xp.resize(self.array, (new_array_size, self.N))
        self.array[current_array_size:] = 0
        self.__notify_observer()

        for free_index in range(current_array_size, self.array.shape[0]):
            heappush(self.free_indices, free_index)

        index = heappop(self.free_indices) # lowest possible index is returned, so the array space is reused effectively
        assert not index in self.leased_indices 
        self.leased_indices.add(index)
        self.max_leased_index = max(self.max_leased_index, index)
        return index

    def lease_many(self, count):
        indices = []
        
        while len(indices) < count:
            remn = count - len(indices)
            assert remn >= 0
            
            if len(self.free_indices) > remn:
                self.free_indices.sort()
                
            inds_batch = self.free_indices[:remn]
            self.free_indices = self.free_indices[remn:]

            if inds_batch:
                len_before = len(self.leased_indices)
                self.leased_indices.update(inds_batch)
                assert len(self.leased_indices) == len_before + len(inds_batch)
                indices.extend(inds_batch)

            if len(indices) < count:
                assert len(self.free_indices) == 0
                
                current_array_size = self.array.shape[0]
                new_array_size = self.grow_policy(current_array_size)
                assert new_array_size > current_array_size
                
                self.array = self.xp.resize(self.array, (new_array_size, self.N))
                self.array[current_array_size:] = 0
                self.__notify_observer()
                self.free_indices.extend(range(current_array_size, self.array.shape[0]))

        assert len(indices) == count
        self.max_leased_index = max(self.leased_indices) if self.leased_indices else -1
        heapify(self.free_indices)
        return indices

    def release(self, index, do_wipeout_data=True):
        assert index in self.leased_indices
        heappush(self.free_indices, index)
        self.leased_indices.discard(index)
        
        if index == self.max_leased_index:
            # max on set of say 50k takes xxx microseconds. So rescan only when max_leased_index was popped!
            self.max_leased_index = max(self.leased_indices) if self.leased_indices else -1

        if do_wipeout_data:
            self.array[index] = 0

    def release_many(self, indices, do_wipeout_data=True):
        indices_set = set(indices)
        assert len(indices_set) == len(indices)
        len_before = len(self.leased_indices)
        self.leased_indices -= indices_set
        assert len(self.leased_indices) == len_before - len(indices)

        self.free_indices.extend(indices)
        heapify(self.free_indices)
        self.max_leased_index = max(self.leased_indices) if self.leased_indices else -1

        if do_wipeout_data:
            self.array[np.array(indices)] = 0

    def clear(self, is_hard_clear=False):
        if is_hard_clear:
            self.array = self.xp.zeros((self.initial_length, self.N), dtype=self.array.dtype)
            self.__notify_observer()
        else:
            self.array[:] = 0
            
        self.free_indices = list(range(self.array.shape[0]))
        heapify(self.free_indices)
        self.leased_indices = set()
        self.max_leased_index = -1

    @property
    def array_active(self):
        return self.array[:self.active_len]

    def __notify_observer(self):
        if not self.observer is None:
            self.observer.size_changed(self.array.shape[0])
            

class DynArrayObserver:
    def size_changed(self, new_size):
        raise NotImplementedError()
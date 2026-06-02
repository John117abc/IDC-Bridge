import numpy as np
import sys

from utils import get_logger
logger = get_logger('per_buffer')

class SumTree:
    """SumTree 数据结构，用于高效按优先级采样"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=np.int32)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)


    def get_leaf(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        return self.tree[0]


class PERBuffer:
    """
    优先经验回放缓冲区。
    经验存储为 (window, world_idx, path_idx)，通过独立 numpy 数组存储。
    """
    def __init__(self, capacity=100000, min_start_train=256,
                 alpha=0.6, beta=0.4, beta_increment=0.001,
                 window_size=16, state_dim=62):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.min_start_train = min_start_train
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.last_indices = None

        self.window_size = window_size
        self.state_dim = state_dim
        self.windows = np.zeros((capacity, window_size, state_dim), dtype=np.float32)
        self._world_idx = np.zeros(capacity, dtype=np.int32)
        self._path_idx = np.zeros(capacity, dtype=np.int32)

    def _calc_initial_priority(self, experience):
        return 1.0

    def handle_new_experience(self, experience, priority=None):
        if priority is None:
            priority = self._calc_initial_priority(experience)
        window, w, p = experience
        self.add(window, w, p, priority)

    def add_safety_trajectory(self, transitions):
        for window, w, p, _ in transitions:
            self.add(window, w, p, 10.0)

    def add(self, window, world_idx, path_idx, priority):
        p = max(priority, 1e-6) ** self.alpha
        ptr = self.tree.data_pointer
        self.windows[ptr] = window
        self._world_idx[ptr] = world_idx
        self._path_idx[ptr] = path_idx
        self.tree.add(p, ptr)
        self.max_priority = max(self.max_priority, p)

    def sample_batch(self, batch_size):
        if self.tree.n_entries < batch_size:
            batch_size = self.tree.n_entries
        if batch_size == 0:
            return (np.zeros((0, self.window_size, self.state_dim), dtype=np.float32),
                    np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))

        batch_windows = np.zeros((batch_size, self.window_size, self.state_dim),
                                  dtype=np.float32)
        batch_worlds = np.zeros(batch_size, dtype=np.int32)
        batch_paths = np.zeros(batch_size, dtype=np.int32)
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data_idx = self.tree.get_leaf(s)
            batch_windows[i] = self.windows[data_idx]
            batch_worlds[i] = self._world_idx[data_idx]
            batch_paths[i] = self._path_idx[data_idx]
            idxs.append(idx)
            priorities.append(p)

        self.last_indices = idxs
        return batch_windows, batch_worlds, batch_paths

    def update_last_batch_priorities(self, new_priorities):
        p_array = np.array(new_priorities, dtype=np.float64)
        if len(np.unique(p_array)) > 1:
            logger.debug(f"update: {len(p_array)} priorities, unique: {len(np.unique(p_array))}")

        if self.last_indices is None:
            return
        p_array = np.array(new_priorities, dtype=np.float64)
        p_array = np.maximum(p_array, 1e-6) ** self.alpha
        for idx, p in zip(self.last_indices, p_array):
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def should_start_training(self):
        return self.tree.n_entries >= self.min_start_train

    def clear(self):
        self.tree = SumTree(self.capacity)
        self.windows = np.zeros((self.capacity, self.window_size, self.state_dim),
                                 dtype=np.float32)
        self._world_idx = np.zeros(self.capacity, dtype=np.int32)
        self._path_idx = np.zeros(self.capacity, dtype=np.int32)
        self.max_priority = 1.0
        self.last_indices = None

    def get_save_buffer_data(self):
        return {
            'tree_data': self.tree.tree.copy(),
            'data': self.tree.data.copy(),
            'data_pointer': self.tree.data_pointer,
            'n_entries': self.tree.n_entries,
            'max_priority': self.max_priority,
            'beta': self.beta,
            'capacity': self.capacity,
            'min_start_train': self.min_start_train,
            'windows': self.windows.copy(),
            '_world_idx': self._world_idx.copy(),
            '_path_idx': self._path_idx.copy(),
        }

    def load_buffer_data(self, data):
        self.tree.tree = data['tree_data'].copy()
        self.tree.data = data['data'].copy()
        self.tree.data_pointer = data['data_pointer']
        self.tree.n_entries = data['n_entries']
        self.max_priority = data.get('max_priority', 1.0)
        self.beta = data.get('beta', self.beta)
        self.capacity = data.get('capacity', self.capacity)
        self.min_start_train = data.get('min_start_train', self.min_start_train)
        if 'windows' in data:
            self.windows = data['windows'].copy()
            self._world_idx = data['_world_idx'].copy()
            self._path_idx = data['_path_idx'].copy()
        else:
            self.windows = self.windows
            self._world_idx = self._world_idx
            self._path_idx = self._path_idx

    def __len__(self):
        return self.tree.n_entries

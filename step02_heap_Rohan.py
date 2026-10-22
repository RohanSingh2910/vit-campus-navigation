"""Step 2 - Rohan: binary min-heap written from scratch (no heapq).

Search inserts tuples (priority, insertion_number, node, path_cost).
The insertion number gives deterministic FIFO ordering for equal priorities.
"""


class MinHeap:
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def push(self, item):
        self.items.append(item)
        child = len(self.items) - 1
        # Only the new leaf can violate the parent <= child invariant.
        while child > 0:
            parent = (child - 1) // 2
            if self.items[parent] <= self.items[child]:
                break
            self.items[parent], self.items[child] = self.items[child], self.items[parent]
            child = parent

    def pop(self):
        if not self.items:
            raise IndexError("Cannot pop an empty heap.")
        smallest = self.items[0]
        last = self.items.pop()
        if self.items:
            self.items[0] = last
            parent = 0
            # Move the replacement root down by swapping with its smaller child.
            while 2 * parent + 1 < len(self.items):
                child = 2 * parent + 1
                right = child + 1
                if right < len(self.items) and self.items[right] < self.items[child]:
                    child = right
                if self.items[parent] <= self.items[child]:
                    break
                self.items[parent], self.items[child] = self.items[child], self.items[parent]
                parent = child
        return smallest


if __name__ == "__main__":
    heap = MinHeap()
    for item in [5, 1, 4, 1, 3]:
        heap.push(item)
    assert [heap.pop() for _ in range(5)] == [1, 1, 3, 4, 5]
    print("Rohan's heap check passed.")

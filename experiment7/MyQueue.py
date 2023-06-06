class MyQueue:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            return None
        return self.items.pop(0)

    def peek(self):
        return self.items[0]

    def getLast(self):
        if self.is_empty():
            return None
        return self.items[len(self.items) - 1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

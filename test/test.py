class A:
    name='chen'
    def __init__(self,name):
        self.name=name

    def get_str(self):
        print("A.name"+self.name)



class B:
    name='yu'
    def __init__(self,name):
        self.name=name

    def get_str(self):
        print("B.name"+self.name)

class C(A,B):
    name='hao'
    def __init__(self, name):
        super().__init__(name)
        self.name=name
    def __init__(self):
      return

    def get_str(self):
        print("C.name"+self.name)

if  __name__ == '__main__':
    c=C();
    c.get_str()
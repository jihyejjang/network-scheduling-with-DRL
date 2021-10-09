from b import B
from dataclasses import dataclass

@dataclass
class Flow:  # type(class1:cc,2:ad,3:vd,4:be),Num,deadline,generate_time,depart_time,bits
    type_: int = None
    num_: int = None
    deadline_: float = None  # millisecond 단위, arrival time - generated time < deadline 이어야 함
    generated_time_: float = None  # millisecond 단위
    node_arrival_time_: list = None
    node_departure_time_: list = None
    arrival_time_: float = None
    bits_: int = None
    met_: bool = None

class A():
    def __init__(self):
        self.b = B()
        self.ex = []

    def pr(self,a):
        f=Flow()
        self.ex.append(self.b.fw(f,a))
        print(self.ex)
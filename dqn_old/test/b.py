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

class B():
    def fw(self,f,a):
        f.type_ = a
        f.num_ = 1
        return f
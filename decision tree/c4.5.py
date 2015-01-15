import copy

class Node:
    def __init__(self):
        self.attrth = 0
        self.attr = 0
        self.next_sibling = None
        self.child = None

def read_data():
    f = open("australian.dat",'r')
    l = f.readlines()
    p = [ tmp.strip().split(" ") for tmp in l]
    ip =[ list(map(float,tmp)) for tmp in p ]
    return ip

def cal_entroy(cur_attr_dict, ip):
    cur_attr = cur_attr_dict.popitem()
    cur_attr_num = cur_attr[0]
    cur_attr_value = cur_attr[1]
    total_sum_zero = 0
    total_sum_one = 0
    for item in ip:
        if item[cur_attr_num] == cur_attr_value:
            if item[-1] > 0:
                total_sum_one += 1
            else:
                total_sum_zero += 1

    return total_sum_zero, total_sum_one


def construct_path(node, done_attr, ip):
    return 0

def main():
    ip = read_data()
    root = Node()
    root.attrth = 1
    print(cal_entroy( {0:0} ,ip ))
    print(root.attrth)

main()

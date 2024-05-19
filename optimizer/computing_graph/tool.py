import socket
from enum import Enum

default_op_stat = {
    'framework_op_stats^': ('out:csv', '', 'op_profile.csv'),
}

default_memory_stat = {
    'memory_profile^': ('', '', 'mem_profile.json')
}

device_name = socket.gethostname()
default_memory_viewer = {
    'memory_viewer^': ('', device_name, 'mem_viewer.json')
}


# Define an enumeration
class CONF(Enum):
    OP = default_op_stat
    MEM = default_memory_stat
    MEM_VIEWER = default_memory_viewer


class Conf_TB:
    def __init__(self, conf: CONF):
        assert len(conf.value) == 1
        for tool, dict_info in conf.value.items():
            self.tool = tool
            self.params = {'tqx': dict_info[0], 'host': dict_info[1]}
            self.output_path = dict_info[2]

    def __str__(self):
        return str(self.tool) + ' ' + str(self.params) + ' ' + str(self.output_path)

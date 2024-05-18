from enum import Enum

default_op_stat = {
    'framework_op_stats^': ('out:csv', 'op_profile.csv'),
}

default_memory_stat = {
    'memory_profile^': ('', 'mem_profile.json')
}


# Define an enumeration
class CONF(Enum):
    OP = default_op_stat
    MEM = default_memory_stat


class Conf_TB:
    def __init__(self, conf: CONF):
        assert len(conf.value) == 1
        for tool, dict_info in conf.value.items():
            self.tool = tool
            self.params = {'tqx': dict_info[0]}
            self.output_path = dict_info[1]

    def __str__(self):
        return str(self.tool) + ' ' + str(self.params) + ' ' + str(self.output_path)


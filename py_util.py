import argparse
import os


from tensorflow.python.framework.dtypes import DType

import tensorflow as tf


def print_communication_cost(table_str):
    # Split the table into lines
    lines = table_str.split('\n')

    # Initialize a list to hold the titles
    filtered_lines = lines[0:3]

    # Search for rows that contain the keyword 'AllReduce'
    for line in lines[3:]:
        if 'all_reduce' in line:
            filtered_lines.append(line)

    filtered_lines = filtered_lines + lines[-3:]

    for line in filtered_lines:
        print(line)


def tensor_shape_to_bits(tensor_shape, dtype: DType):
    """
    Convert a TensorShape to the total number of bits.

    :param tensor_shape: A TensorShape object or a list/tuple of dimensions.
    :param dtype: The data type of the tensor elements (e.g., tf.float32, tf.float64).
    :return: Total size in bits.
    """
    # Get the total number of elements
    if None in tensor_shape:
        return 0
    total_elements = tf.reduce_prod(tensor_shape).numpy()

    # Determine the number of bits per element based on dtype
    if dtype in [tf.float32, tf.int32, tf.uint32]:
        bits_per_element = 32
    elif dtype in [tf.float64, tf.int64, tf.uint64, tf.complex64]:
        bits_per_element = 64
    elif dtype in [tf.float16, tf.int16, tf.uint16]:
        bits_per_element = 16
    elif dtype in [tf.int8, tf.uint8]:
        bits_per_element = 8
    elif dtype == tf.bool:
        bits_per_element = 1
    elif dtype == tf.resource:
        # Hypothetical placeholder size for resource handles
        bits_per_element = 0  # Example placeholder, change as appropriate
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    # Calculate the total size in bits
    total_bits = int(total_elements * bits_per_element)

    return total_bits


def convert_data_size(value, from_unit, to_unit):
    """
    Convert between different data size units.

    :param value: The numerical value to convert.
    :param from_unit: The unit of the input value (e.g., 'GB', 'Mb', 'bytes', etc.).
    :param to_unit: The unit to convert to (e.g., 'GB', 'Mb', 'bytes', etc.).
    :return: The converted value.
    """

    # Define conversion factors relative to bytes
    conversion_factors = {
        'bit': 1 / 8,  # 1 bit is 1/8 bytes
        'byte': 1,  # 1 byte is 1 byte
        'B': 1,  # 1 byte is 1 byte
        'KB': 1000,  # 1 kilobyte is 1024 bytes
        'MB': 1000 ** 2,  # 1 megabyte is 1024^2 bytes
        'GB': 1000 ** 3,  # 1 gigabyte is 1024^3 bytes
        'TB': 1000 ** 4,  # 1 terabyte is 1024^4 bytes
        'PB': 1000 ** 5,  # 1 petabyte is 1024^5 bytes
        'kbit': 1000 / 8,  # 1 kilobit (kbit) is 1000 bits or 1000/8 bytes
        'Mbit': (1000 ** 2) / 8,  # 1 megabit (Mbit) is 1000^2 bits or 1000000/8 bytes
        'Gbit': (1000 ** 3) / 8,  # 1 gigabit (Gbit) is 1000^3 bits or 1000000000/8 bytes
        'Tbit': (1000 ** 4) / 8,  # 1 terabit (Tbit) is 1000^4 bits or 1000000000000/8 bytes
        'Pbit': (1000 ** 5) / 8  # 1 petabit (Pbit) is 1000^5 bits or 1000000000000000/8 bytes
    }

    # Convert the input value to bytes
    bytes_value = value * conversion_factors[from_unit]

    # Convert the bytes value to the target unit
    converted_value = bytes_value / conversion_factors[to_unit]

    return converted_value


def convert_time(value, from_unit, to_unit):
    """
    Convert between different time units.

    :param value: The numerical value to convert.
    :param from_unit: The unit of the input value (e.g., 's', 'ms', 'min', etc.).
    :param to_unit: The unit to convert to (e.g., 's', 'ms', 'min', etc.).
    :return: The converted value.
    """

    # Define conversion factors relative to seconds
    conversion_factors = {
        'ns': 1e-9,  # Nanoseconds to seconds
        'µs': 1e-6,  # Microseconds to seconds
        'us': 1e-6,  # Microseconds to seconds (alternative symbol)
        'ms': 1e-3,  # Milliseconds to seconds
        's': 1,  # Seconds to seconds
        'min': 60,  # Minutes to seconds
        'h': 3600,  # Hours to seconds
        'd': 86400,  # Days to seconds
        'w': 604800  # Weeks to seconds
    }

    # Convert the input value to seconds
    seconds_value = value * conversion_factors[from_unit]

    # Convert the seconds value to the target unit
    converted_value = seconds_value / conversion_factors[to_unit]

    return converted_value


# Compare
def compare_2d_list(list1, list2):
    def normalize_2d_list(lst):
        return sorted([sorted(sublist) for sublist in lst])

    return normalize_2d_list(list1) == normalize_2d_list(list2)

# import ConfigParser
#
# class mdp_config(object):
#   config = ConfigParser.RawConfigParser()
#   config.read('util.config')


LABEL_MAPPING_3_CLASS = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 1,
    10: 1,
    11: 2,
    12: 2,
    13: 0,
}

LABEL_MAPPING_2_CLASS = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 0,
}

LABEL_MAPPING_2_CLASS2 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 0,
    10: 0,
    11: 1,
    12: 1,
    13: 0,
}

MAPPING_LABEL = {0: '',
                 1: 'Opening',
                 2: 'Closing',
                 3: 'Entering',
                 4: 'Exiting',
                 5: 'Open_Trunk_car',
                 6: 'Closing_Trunk_car',
                 7: 'Open_Trunk_pickup',
                 8: 'Closing_Trunk_pickup',
                 9: 'Loading_door',
                 10: 'Unloading_door',
                 11: 'Loading_trunk',
                 12: 'Unloading_trunk'}

MAPPING_13_TO_9 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,  # open trunk
    6: 6,  # close trunk
    7: 5,
    8: 6,
    9: 0,
    10: 0,
    11: 7,
    12: 8,
    13: 0,
}

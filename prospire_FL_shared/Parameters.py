import math
import warnings
warnings.filterwarnings("ignore")


class Parameters:

    def __init__(self, dataset):
        self.dataset = dataset

        if dataset == 'wi_outdoor_15_buildings_8_reflections':
            self.power_flies_dir = 'WIOutdoor15Buildings8ReflectionsRxPowerFiles/'
            self.tx_locs_dir = 'WIOutdoor15Buildings8ReflectionsTxRxLocations/txset.txrx'
            self.rx_locs_dir = 'WIOutdoor15Buildings8ReflectionsTxRxLocations/rxset.txrx'
            self.buildings_file = 'WIOutdoor15Buildings8ReflectionsBuildingVertices/ConcreteBuildings.object'
            self.foliage_file = None
            self.xy_grid_rx = True
            self.x_min, self.y_min, self.x_max, self.y_max = 0.0, 0.0, 500.0, 500.0
            self.cell_width = 10.0
            # Total number of TXs is 152 and RXs is 2340
            self.num_sensors_list = range(10, 315, 50)
            # self.num_train_tx_pos_list = range(10, 95, 20)
            self.num_train_tx_pos_list = range(70, 75, 20)
            self.num_train_tx_pos = 90
            self.num_sensors = 200
            self.num_exp_test_tx = 30  # Number of experimental TX positions to test
            self.num_target_locs_for_radio_map = 200
            self.rss_min = -109.0  # Minimum value of acceptable rss
            self.rss_max = -2.0  # Maximum value of acceptable rss
            self.valid_rss = self.rss_min
            self.clip_low_rss = True













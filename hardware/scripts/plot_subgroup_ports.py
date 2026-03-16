#!/usr/bin/env python3
import sys

import plot_port_utilization


if __name__ == "__main__":
    plot_port_utilization.main([*sys.argv[1:], "--boundary", "subgroup"])
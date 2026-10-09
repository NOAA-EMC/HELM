"""Build a spherical cap with periodic rings and one shared North Pole node."""

import numpy as np


def polar_coordinates(description):
    longitude_count = int(description["n"])
    radial_intervals = int(description["m"])
    minimum_latitude = float(description.get("minimum_latitude", 60))
    rotation = float(description.get("rotation", 0))
    if longitude_count < 3 or radial_intervals < 1 or not 0 < minimum_latitude < 90:
        raise ValueError("Polar meshes require n >= 3, m >= 1 and 0 < minimum_latitude < 90")
    longitudes = rotation + np.arange(longitude_count) * 360 / longitude_count
    latitudes = np.linspace(minimum_latitude, 90, radial_intervals + 1)[:-1]
    longitude = np.tile(longitudes, radial_intervals)
    longitude = (longitude + 180) % 360 - 180
    latitude = np.repeat(latitudes, longitude_count)
    pole = len(longitude)
    longitude = np.append(longitude, 0)
    latitude = np.append(latitude, 90)
    cells, parents = [], []
    for ring in range(radial_intervals - 1):
        for column in range(longitude_count):
            following = (column + 1) % longitude_count
            outer = ring * longitude_count
            inner = outer + longitude_count
            quad = [outer + column, outer + following, inner + following, inner + column]
            cells.extend([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
            parents.extend([len(parents) // 2 + 1] * 2)
    for column in range(longitude_count):
        start = (radial_intervals - 1) * longitude_count
        cells.append([start + column, start + (column + 1) % longitude_count, pole])
        parents.append(longitude_count * (radial_intervals - 1) + column + 1)
    return longitude, latitude, cells, parents

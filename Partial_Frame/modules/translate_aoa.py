"""
Copyright (C) 2024 Khandaker Foysal Haque
contact: haque.k@northeastern.edu
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import math
import numpy as np


# Please refer to Figure of the paper SAWEC for better understanding of the function
def translate_aoa_to_360_frame_angle(Ax, Bx, AC, theta_tx, aoa):
    OA = AC / math.cos(np.radians(aoa))

    # Calculate OC, CD, and BD
    OC = abs(OA * np.sin(np.radians(aoa)))
    CD = np.abs(Ax - Bx)
    BD = AC

    # Determine the sign of ζpos and ζaoa
    zeta_pos = np.sign(Ax - Bx)
    zeta_aoa = np.sign(aoa)

    # Calculate OD
    OD = OC + zeta_pos * zeta_aoa * CD

    # Calculate ∠OBD in radians::::: Compensated aoa
    aoa_frame_rad = math.atan(OD / BD)

    # Convert angle to degrees:::::this is compensated aoa for colocation distance
    aoa_frame_deg = np.degrees(aoa_frame_rad) * zeta_aoa

    # Calculate OB using the law of sines
    OB = abs(BD / np.cos(aoa_frame_deg))

    # Translate AoA to 360 Frame Scale
    theta = (aoa_frame_deg + theta_tx) % 360
    return aoa_frame_deg, OB, theta

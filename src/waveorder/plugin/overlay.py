"""The retardance and orientation overlay of the Visualization tab: orientation as hue, retardance as value."""

import numpy as np


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Return the RGB colors of HSV colors in ``[0, 1]``, both with the color last (``(..., 3)``)."""
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sector = np.floor(hue * 6)
    fraction = hue * 6 - sector
    sector = sector.astype(int) % 6
    p = value * (1 - saturation)
    q = value * (1 - saturation * fraction)
    t = value * (1 - saturation * (1 - fraction))
    red = np.choose(sector, [value, q, p, p, t, value])
    green = np.choose(sector, [t, value, value, q, p, p])
    blue = np.choose(sector, [p, p, t, value, value, q])
    return np.stack((red, green, blue), axis=-1)


def ret_ori_overlay(czyx: np.ndarray, ret_max: float = 10) -> np.ndarray:
    """Return the HSV overlay of retardance and orientation: orientation as hue, retardance as value.

    Parameters
    ----------
    czyx : numpy.ndarray
        ``czyx[0]`` is the retardance in nanometers, ``czyx[1]`` the orientation in radians in ``[0, pi]``.
    ret_max : float
        Retardance shown at full value; larger retardance is clipped.

    Returns
    -------
    numpy.ndarray
        RGB image of shape ``(3, ...)``.

    Raises
    ------
    ValueError
        If ``czyx`` does not have two channels.
    """
    if czyx.shape[0] != 2:
        raise ValueError(f"Input must have shape (2, ...) instead of ({czyx.shape[0]}, ...)")
    retardance = np.clip(czyx[0], 0, ret_max)
    largest = np.max(retardance)
    value = retardance / largest if largest > 0 else np.zeros_like(retardance)
    hsv = np.stack((czyx[1] / np.pi % 1, np.ones_like(retardance), value), axis=-1)
    return np.moveaxis(hsv_to_rgb(hsv).astype(np.float32), -1, 0)

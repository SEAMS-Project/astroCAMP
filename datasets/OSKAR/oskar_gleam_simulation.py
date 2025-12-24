#!/usr/bin/env python3
import os
import sys
import logging
import argparse

import numpy as np
import oskar

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.io import fits


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Running OSKAR simulation")

    parser.add_argument("--telescope_lon", required=True, type=float,
                        help="Longitude of telescope [deg]")
    parser.add_argument("--telescope_lat", required=True, type=float,
                        help="Latitude of telescope [deg]")
    parser.add_argument("--fov_deg", type=float,
                        help="Field of view [degree]")
    parser.add_argument("--num_time_steps", required=True, type=int,
                        help="Number of time steps")
    parser.add_argument("--input_directory", required=True,
                        help=".tm input directory")
    parser.add_argument("--phase_centre_ra_deg", required=True, type=float,
                        help="Phase centre right ascension [deg]")
    parser.add_argument("--phase_centre_dec_deg", required=True, type=float,
                        help="Phase centre declination [deg]")
    parser.add_argument("--out_name", required=True,
                        help="Output name")
    parser.add_argument("--precision", required=True,
                        choices=["single", "double"],
                        help="OSKAR precision")
    parser.add_argument("--start_frequency_hz", required=True, type=float,
                        help="Start frequency in Hz")
    parser.add_argument("--num_channels", type=int, default=1,
                        help="Number of channels")
    parser.add_argument("--frequency_inc_hz", type=float, default=20e6,
                        help="Frequency increment in Hz")
    parser.add_argument("--use_gpus", action="store_true",
                        help="Enable GPU usage")
    parser.add_argument("--length_sec", required=True, type=int,
                        help="Observation length [sec]")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    log = logging.getLogger()
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    return log


# -----------------------------------------------------------------------------
# Time / geometry helpers
# -----------------------------------------------------------------------------
def compute_observation_times(
    telescope: EarthLocation,
    target: SkyCoord,
    length_sec: int,
) -> tuple[Time, Time]:
    """
    Returns (t_start, t_transit)
    """
    t_guess = Time("2025-11-10 00:00:00", scale="utc")
    lst_guess = t_guess.sidereal_time(
        "apparent", longitude=telescope.lon
    )
    ha = lst_guess - target.ra

    # Convert degrees to hours
    t_transit = t_guess - (ha / (15 * u.deg / u.hour))

    # Round transit time to closest 30 minutes
    half_hour = 30 * 60
    sec_rounded = np.round(t_transit.unix / half_hour) * half_hour
    t_rounded = Time(sec_rounded, format="unix", scale="utc")

    td_half_length = TimeDelta(length_sec / 2, format="sec")
    t_start = t_rounded - td_half_length

    return t_start, t_transit


# -----------------------------------------------------------------------------
# Sky model construction
# -----------------------------------------------------------------------------
def build_gleam_sky(
    args: argparse.Namespace,
    log: logging.Logger,
) -> oskar.Sky:
    sky = oskar.Sky(args.precision)

    hdulist = fits.open("GLEAM_EGC_v2.fits")
    cols = hdulist[1].data[0].array

    data = np.column_stack(
        (cols["RAJ2000"], cols["DEJ2000"], cols["peak_flux_wide"])
    )
    data = data[data[:, 2].argsort()[::-1]]

    log.info("Loaded %d GLEAM sources", data.shape[0])

    sky_gleam = oskar.Sky.from_array(data, args.precision)
    sky.append(sky_gleam)

    sky.filter_by_radius(
        0.0,
        np.sqrt(2) * args.fov_deg / 2,
        args.phase_centre_ra_deg,
        args.phase_centre_dec_deg,
    )
    sky.filter_by_flux(1, 5)

    log.info(
        "Number of sources in inner sky model: %d",
        sky.num_sources,
    )

    return sky


# -----------------------------------------------------------------------------
# OSKAR settings
# -----------------------------------------------------------------------------
def build_settings(
    args: argparse.Namespace,
    t_transit: Time,
) -> oskar.SettingsTree:
    params = {
        "simulator": {
            "use_gpus": bool(args.use_gpus),
            "keep_log_file": True,
        },
        "observation": {
            "num_channels": args.num_channels,
            "start_frequency_hz": args.start_frequency_hz,
            "frequency_inc_hz": args.frequency_inc_hz,
            "phase_centre_ra_deg": args.phase_centre_ra_deg,
            "phase_centre_dec_deg": args.phase_centre_dec_deg,
            "num_time_steps": args.num_time_steps,
            "start_time_utc": t_transit.utc.isot,
            "length": args.length_sec,
        },
        "telescope": {
            "input_directory": args.input_directory,
            "station_type": "Isotropic",
        },
        "interferometer": {
            "ms_filename": args.out_name + ".ms",
            "channel_bandwidth_hz": 1e6,
            "time_average_sec": 10,
        },
    }

    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(params)

    if args.precision == "single":
        settings["simulator/double_precision"] = False

    return settings


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"-I- OSKAR version = {oskar.__version__}")

    args = parse_args()
    print("oskar_gleam_simulation.py args =", args)

    log = setup_logging()

    telescope = EarthLocation(
        lon=args.telescope_lon * u.deg,
        lat=args.telescope_lat * u.deg,
        height=0,
    )
    log.info("Telescope = %s", telescope)

    target = SkyCoord(
        ra=args.phase_centre_ra_deg * u.deg,
        dec=args.phase_centre_dec_deg * u.deg,
    )
    log.info("Target = %s", target)

    t_start, t_transit = compute_observation_times(
        telescope, target, args.length_sec
    )
    log.info("Transit time (UTC): %s", t_transit.utc.isot)
    log.info(
        "Start time (UTC): %s for duration %d sec",
        t_start.utc.isot,
        args.length_sec,
    )

    sky = build_gleam_sky(args, log)
    settings = build_settings(args, t_transit)

    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(sky)

    log.info("Starting OSKAR simulation")
    sim.run()
    log.info("Simulation finished")


if __name__ == "__main__":

    main()

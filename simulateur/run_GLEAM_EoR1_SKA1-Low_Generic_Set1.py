#!/usr/bin/env python3

"""Generate some initial simulated datasets (SIM-147 and SIM-197)."""

import logging
import sys

from astropy.io import fits
from astropy.time import Time, TimeDelta
import numpy
import oskar


LOG = logging.getLogger()


def get_start_time(ra0_deg, length_sec):
    """Returns optimal start time for field RA and observation length."""
    t = Time('2000-01-01 00:00:00', scale='utc', location=('116.764d', '0d'))
    dt_hours = 24.0 - t.sidereal_time('apparent').hour + (ra0_deg / 15.0)
    start = t + TimeDelta(dt_hours * 3600.0 - length_sec / 2.0, format='sec')
    return start.value


def make_sky_model(sky0, settings, radius_deg):
    """Filter sky model. Includes all sources within the given radius."""
    # Get pointing centre.
    ra0_deg = float(settings['observation/phase_centre_ra_deg'])
    dec0_deg = float(settings['observation/phase_centre_dec_deg'])

    # Create filtered sky model.
    sky = sky0.create_copy()
    sky.filter_by_radius(0.0, radius_deg, ra0_deg, dec0_deg)
    LOG.info("Number of sources in input sky model: %d", sky0.num_sources)
    LOG.info("Number of sources in filtered sky model: %d", sky.num_sources)
    return sky


def make_vis_data(name, sky0, ra_deg, dec_deg, max_radius_deg, length_sec,
                  num_time_steps, num_channels):
    """Generates visibility data."""
    settings_dict = {
        'simulator': {
            'max_sources_per_chunk': '2000'
        },
        'observation' : {
            'length': str(length_sec),
            'num_time_steps': str(num_time_steps),
            'start_frequency_hz': '140e6',
            'frequency_inc_hz': '1e6',
            'num_channels': str(num_channels),
            'phase_centre_ra_deg': str(ra_deg),
            'phase_centre_dec_deg': str(dec_deg)
        },
        'telescope': {
            'input_directory': 'SKA1-LOW_SKO-0000422_Rev3_38m.tm',
            'pol_mode': 'Scalar'
        },
        'interferometer': {
            'channel_bandwidth_hz': '100e3',
            'time_average_sec': '1.0',
            'max_time_samples_per_block': '4',
            'ms_filename': 'sim_' + name + '.ms',
            'oskar_vis_filename': 'sim_' + name + '.vis'
        }
    }
    settings = oskar.SettingsTree('oskar_sim_interferometer')
    settings.from_dict(settings_dict)
    settings['observation/start_time_utc'] = get_start_time(ra_deg, length_sec)
    sky = make_sky_model(sky0, settings, max_radius_deg)
    sky.save('sky_' + name + '.txt')
    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(sky)
    sim.run()


def make_image(name, algorithm, root_prefix, fov_deg, num_pixels):
    """Generates an image from visibility data."""
    settings_dict = {
        'image': {
            'size': str(num_pixels),
            'fov_deg': str(fov_deg),
            'algorithm': algorithm,
            'fft/use_gpu': 'true',
            'fft/grid_on_gpu': 'true',
            'input_vis_data': 'sim_' + name + '.vis',
            'root_path': root_prefix + '_' + name
        }
    }
    settings = oskar.SettingsTree('oskar_imager')
    settings.from_dict(settings_dict)
    if name == 'large':
        settings['image/fft/use_gpu'] = 'false'
    LOG.info('Starting imager for "%s"', settings['image/root_path'])
    imager = oskar.Imager(settings=settings)
    imager.run()
    LOG.info('Imaging complete')


def main():
    """Main function."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)

    # Sky model from GLEAM catalogue (Hurley-Walker et al. 2017).
    sky0 = oskar.Sky()
    hdulist = fits.open('GLEAM_EGC.fits')
    # pylint: disable=no-member
    cols = hdulist[1].data[0].array
    data = numpy.column_stack(
        (cols['RAJ2000'], cols['DEJ2000'], cols['peak_flux_wide']))
    data = data[data[:, 2].argsort()[::-1]]
    sky_gleam = oskar.Sky.from_array(data)
    sky0.append(sky_gleam)

    # Make visibility data.
    make_vis_data('large', sky0, 60.0, -30.0, 5.657, 14400.0, 60, 4)
    make_vis_data('medium', sky0, 60.0, -30.0, 2.828, 3600.0, 30, 2)
    make_vis_data('small', sky0, 60.0, -30.0, 0.707, 900.0, 30, 1)

    # Make images.
    make_image('large', 'W-projection', 'img_wproj', 8.0, 16384)
    make_image('medium', 'W-projection', 'img_wproj', 4.0, 8192)
    make_image('small', 'W-projection', 'img_wproj', 1.0, 2048)
    # make_image('large', 'DFT 3D', 'img_dft', 8.0, 16384)
    # make_image('medium', 'DFT 3D', 'img_dft', 4.0, 8192)
    # make_image('small', 'DFT 3D', 'img_dft', 1.0, 2048)

if __name__ == '__main__':
    main()

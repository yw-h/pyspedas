from pyspedas.mms.mms_load_data import mms_load_data
from pyspedas.mms.feeps.mms_feeps_correct_energies import mms_feeps_correct_energies
from pyspedas.mms.feeps.mms_feeps_flat_field_corrections import mms_feeps_flat_field_corrections
from pyspedas.mms.feeps.mms_feeps_active_eyes import mms_feeps_active_eyes
from pyspedas.mms.feeps.mms_feeps_split_integral_ch import mms_feeps_split_integral_ch
from pyspedas.mms.feeps.mms_feeps_remove_bad_data import mms_feeps_remove_bad_data
from pyspedas.mms.feeps.mms_feeps_remove_sun import mms_feeps_remove_sun
from pyspedas.mms.feeps.mms_feeps_omni import mms_feeps_omni
from pyspedas.mms.feeps.mms_feeps_spin_avg import mms_feeps_spin_avg
from pyspedas.mms.print_vars import print_vars
from pyspedas.mms.mms_config import CONFIG
from pytplot import time_clip as tclip


@print_vars
def mms_load_feeps(trange=['2015-10-16', '2015-10-17'], probe='1', data_rate='srvy', 
    level='l2', datatype='electron', varformat=None, varnames=[], get_support_data=True, suffix='', time_clip=False,
    no_update=False, available=False, notplot=False, no_flatfield_corrections=False, data_units=['count_rate', 'intensity'], 
    latest_version=False, major_version=False, min_version=None, cdf_version=None, spdf=False, always_prompt=False):
    """
    Load data from the Fly's Eye Energetic Particle Sensor (FEEPS)
    
    Parameters
    ----------
        trange : list of str
            time range of interest [start time, end time] with the format
            'YYYY-MM-DD','YYYY-MM-DD'] or to specify more or less than a day 
            ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']

        probe : str or list of str
            list of probes, valid values for MMS probes are ['1','2','3','4']. 

        data_rate : str or list of str
            instrument data rates for FEEPS include ['brst', 'srvy']. The
            default is 'srvy'.

        level : str
            indicates level of data processing. the default if no level is specified is 'l2'

        datatype : str or list of str
            Valid datatypes for FEEPS are: 
                       L2, L1b: ['electron', 'ion']
                       L1a: ['electron-bottom', 'electron-top', 'ion-bottom', 'ion-top']

        get_support_data: bool
            Data with an attribute "VAR_TYPE" with a value of "support_data"
            will be loaded into tplot.  By default, only loads in data with a 
            "VAR_TYPE" attribute of "data".

        time_clip: bool
            Data will be clipped to the exact trange specified by the trange keyword.
            
        varformat: str
            The file variable formats to load into tplot.  Wildcard character
            "*" is accepted.  By default, all variables are loaded in.

        varnames: list of str
            List of variable names to load (if not specified,
            all data variables are loaded)

        suffix: str
            The tplot variable names will be given this suffix.  By default, 
            no suffix is added.

        notplot: bool
            If True, then data are returned in a hash table instead of 
            being stored in tplot variables (useful for debugging, and
            access to multidimensional data products)

        available: bool
            If True, simply return the available data files (without downloading)
            for the requested parameters

        no_update: bool
            Set this flag to preserve the original data. if not set and newer 
            data is found the existing data will be overwritten

        cdf_version: str
            Specify a specific CDF version # to load (e.g., cdf_version='4.3.0')

        min_version: str
            Specify a minimum CDF version # to load

        latest_version: bool
            Only grab the latest CDF version in the requested time interval

        major_version: bool
            Only open the latest major CDF version (e.g., X in vX.Y.Z) in the requested time interval

        always_prompt: bool
            Set this keyword to always prompt for the user's username and password;
            useful if you accidentally save an incorrect password, or if your SDC password has changed

        spdf: bool
            If True, download the data from the SPDF instead of the SDC

    Returns
    ---------
        List of tplot variables created.

    """

    # as of 3 July 2023, there's a mixture of v7.x.x and v6.x.x files at the SDC
    # these files aren't compatible, so we need to only load the latest major version
    # to avoid crashes (unless otherwise specified)
    if not latest_version and not major_version and min_version is None and cdf_version is None:
        major_version = True

    tvars = mms_load_data(trange=trange, notplot=notplot, probe=probe, data_rate=data_rate, level=level, instrument='feeps',
            datatype=datatype, varformat=varformat, varnames=varnames, get_support_data=get_support_data, suffix=suffix,
            no_update=no_update, available=available, latest_version=latest_version,
            major_version=major_version, min_version=min_version, cdf_version=cdf_version, spdf=spdf, always_prompt=always_prompt)

    if tvars == [] or available or notplot or CONFIG['download_only'] or tvars is None:
        return tvars

    probes = probe if isinstance(probe, list) else [probe]
    data_rates = data_rate if isinstance(data_rate, list) else [data_rate]
    levels = level if isinstance(level, list) else [level]
    datatypes = datatype if isinstance(datatype, list) else [datatype]
    data_units = data_units if isinstance(data_units, list) else [data_units]

    probes = [str(p) for p in probes]

    mms_feeps_correct_energies(probes, data_rate, level=level, suffix=suffix)

    if not no_flatfield_corrections:
        mms_feeps_flat_field_corrections(probes=probes, data_rate=data_rate, suffix=suffix)

    for probe in probes:
        for lvl in levels:
            for drate in data_rates:
                for datatype in datatypes:
                    mms_feeps_remove_bad_data(trange=trange, probe=probe, data_rate=drate, datatype =datatype, level=lvl, suffix=suffix)

                    for data_unit in data_units:
                        eyes = mms_feeps_active_eyes(trange, probe, drate, datatype, lvl)

                        split_vars = mms_feeps_split_integral_ch(data_unit, datatype, probe, suffix=suffix, data_rate=drate, level=lvl, sensor_eyes=eyes)

                        sun_removed_vars = mms_feeps_remove_sun(eyes, trange, probe=probe, datatype=datatype, data_units=data_unit, data_rate=drate, level=lvl, suffix=suffix)

                        omni_vars = mms_feeps_omni(eyes, probe=probe, datatype=datatype, data_units=data_unit, data_rate=drate, level=lvl, suffix=suffix)

                        if split_vars is not None:
                            tvars = tvars + split_vars

                        if sun_removed_vars is not None:
                            tvars = tvars + sun_removed_vars

                        if omni_vars is not None:
                            tvars = tvars + omni_vars

                        spin_avg_vars = mms_feeps_spin_avg(probe=probe, data_units=data_unit, datatype=datatype, data_rate=drate, level=lvl, suffix=suffix)

                        if spin_avg_vars is not None:
                            tvars.append(spin_avg_vars)

    if time_clip:
        for new_var in tvars:
            tclip(new_var, trange[0], trange[1], suffix='')

    return tvars

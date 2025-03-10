from pyspedas.utilities.dailynames import dailynames
from pyspedas.utilities.download import download
from pytplot import time_clip as tclip
from pytplot import cdf_to_tplot
from .rfs import rfs_variables_to_load
from .config import CONFIG

def load(trange=['2018-11-5', '2018-11-6'],
         instrument='fields',
         datatype='mag_RTN',
         spec_types=None,  # for DFB AC spectral data
         level='l2',
         suffix='',
         get_support_data=False,
         varformat=None,
         varnames=[],
         downloadonly=False,
         notplot=False,
         no_update=False,
         time_clip=False,
         username=None,
         password=None,
         last_version=False):
    """
    This function loads Parker Solar Probe data into tplot variables; this function is not 
    meant to be called directly; instead, see the wrappers: 
        psp.fields: FIELDS data
        psp.spc: Solar Probe Cup data
        psp.spe: SWEAP/SPAN-e data
        psp.spi: SWEAP/SPAN-i data
        psp.epihi: ISoIS/EPI-Hi data
        psp.epilo: ISoIS/EPI-Lo data
        psp.epi ISoIS/EPI (merged Hi-Lo) data
    
    """
    # remote path formats generally are going to be all lowercase except for
    # on the Berkeley FIELDS server
    if (username is not None) and (datatype in ['mag_RTN_1min',
                                            'mag_RTN_4_Sa_per_Cyc',
                                            'mag_RTN',
                                            'mag_SC',
                                            'mag_SC_1min',
                                            'mag_SC_4_Sa_per_Cyc',
                                            'sqtn_rfs_V1V2'
                                            ]):
        pass
    else:
        datatype = datatype.lower()

    prefix = 'psp_'  #To cover the case if one *does* call this routine directly.

    file_resolution = 24*3600.
    if instrument == 'fields':
        prefix = '' #CDF Variables are already prefixed with psp_fld_

        # 4_per_cycle and 1min are daily, not 6h like the full resolution 'mag_(rtn|sc)'
        if datatype in ['mag_rtn', 'mag_sc']:
            pathformat = instrument + '/' + level + '/' + datatype + '/%Y/psp_fld_' + level + '_' + datatype + '_%Y%m%d%H_v??.cdf'
            file_resolution = 6*3600.
        elif datatype in ['mag_rtn_1min', 'mag_sc_1min', 'rfs_hfr', 'rfs_lfr', 'rfs_burst', 'f2_100bps', 'aeb']:
            pathformat = instrument + '/' + level + '/' + datatype + '/%Y/psp_fld_' + level + '_' + datatype + '_%Y%m%d_v??.cdf'
        elif datatype in ['mag_rtn_4_per_cycle', 'mag_rtn_4_sa_per_cyc']:
            pathformat = instrument + '/' + level + '/mag_rtn_4_per_cycle/%Y/psp_fld_' + level + '_mag_rtn_4_sa_per_cyc_%Y%m%d_v??.cdf'
        elif datatype in ['mag_sc_4_per_cycle', 'mag_sc_4_sa_per_cyc']:
            pathformat = instrument + '/' + level + '/mag_sc_4_per_cycle/%Y/psp_fld_' + level + '_mag_sc_4_sa_per_cyc_%Y%m%d_v??.cdf'
        elif datatype == 'sqtn_rfs_v1v2':
            pathformat = instrument + '/' + level + '/' + datatype + '/%Y/psp_fld_' + level + '_' + datatype + '_%Y%m%d_v?.?.cdf'        
        elif datatype == 'rfs_lfr_qtn':
            pathformat = instrument + '/' + level + '/' + datatype + '/%Y/psp_fld_' + level + '_' + datatype + '_%Y%m*_v??.cdf'
        elif datatype in ['dfb_dc_spec', 'dfb_ac_spec', 'dfb_dc_xspec', 'dfb_ac_xspec']:
            out_vars = []
            for item in spec_types:
                loaded_data = load(trange=trange, instrument=instrument, datatype=datatype + '_' + item, level=level, 
                    suffix=suffix, get_support_data=get_support_data, varformat=varformat, varnames=varnames, 
                    downloadonly=downloadonly, notplot=notplot, time_clip=time_clip, no_update=no_update, last_version=last_version)
                if loaded_data != []:
                    out_vars.extend(loaded_data)
            return out_vars
        elif datatype[:12] == 'dfb_dc_spec_' or datatype[:12] == 'dfb_ac_spec_' or datatype[:13] == 'dfb_dc_xspec_' or datatype[:13] == 'dfb_ac_xspec_':
            if datatype[:13] == 'dfb_dc_xspec_' or datatype[:13] == 'dfb_ac_xspec_':
                dtype_tmp = datatype[:12]
                stype_tmp = datatype[13:]
            else:
                dtype_tmp = datatype[:11]
                stype_tmp = datatype[12:]
            pathformat = instrument + '/' + level + '/' + dtype_tmp + '/' + stype_tmp + '/%Y/psp_fld_' + level + '_' + datatype + '_%Y%m%d_v??.cdf'
        elif datatype == 'sqtn_rfs_v1v2':
            # unfortunately the naming format of quasi-thermal-noise cdf file is different from others
            pathformat = instrument + '/' + level + '/' + datatype + '/%Y/psp_fld_' + level + '_' + datatype + '_%Y%m%d_v?.?.cdf'
        elif datatype == 'sqtn_rfs_V1V2':
            # unpublished QTN data
            pathformat = instrument + '/' + level + '/' + datatype + '/%Y/%m/psp_fld_' + level + '_' + datatype + '_%Y%m%d_v?.?.cdf'
        elif datatype == 'merged_scam_wf':
            if username == None:
                pathformat = instrument + '/' + level + '/' + datatype + '/%Y/psp_fld_' + level + '_' + datatype + '_%Y%m%d%H_v??.cdf'
            else:
                pathformat = instrument + '/' + level + '/' + datatype + '/%Y/%m/psp_fld_' + level + '_' + datatype + '_%Y%m%d%H_v??.cdf'

        # unpublished data
        elif username != None:
            if datatype in ['mag_RTN', 'mag_SC']:
                pathformat = instrument + '/' + level + '/' + datatype + '/%Y/%m/psp_fld_' + level + '_' + datatype + '_%Y%m%d%H_v??.cdf'
                file_resolution = 6*3600.

            elif datatype in ['mag_RTN_1min', 'mag_RTN_4_Sa_per_Cyc', 'mag_SC_1min', 'mag_SC_4_Sa_per_Cyc']:
                pathformat = instrument + '/' + level + '/' + datatype + '/%Y/%m/psp_fld_' + level + '_' + datatype + '_%Y%m%d_v??.cdf'

            elif datatype ==  'sqtn_rfs_V1V2':
                pathformat = instrument + '/' + level + '/' + datatype + '/%Y/%m/psp_fld_' + level + '_' + datatype + '_%Y%m%d_v?.?.cdf'
            elif datatype in ['ephem_spp_rtn']:
                pathformat = instrument + '/' + level + '/' + datatype + '/%Y/%m/spp_fld_' + level + '_' + datatype + '_%Y%m%d_v01.cdf'
            else:
                pathformat = instrument + '/' + level + '/' + datatype + '/%Y/%m/psp_fld_' + level + '_' + datatype + '_%Y%m%d_v??.cdf'


        else:
            # Generic SPDF path.  
            pathformat = instrument + '/' + level + '/' + datatype + '/%Y/psp_fld_' + level + '_' + datatype + '_%Y%m%d%H_v??.cdf'
            file_resolution = 6*3600.

        # Files on Berkeley server are stored in monthly directories 
        if username != None:
            pathformat = pathformat.replace('/%Y/psp_fld', '/%Y/%m/psp_fld')
            if level == 'l1':
                pathformat = pathformat.replace('psp_fld', 'spp_fld')


    elif instrument == 'spc':
        if username is None:
            prefix = 'psp_spc_'
            pathformat = 'sweap/spc/' + level + '/' + datatype + '/%Y/psp_swp_spc_' + datatype + '_%Y%m%d_v??.cdf'
        else:
            # unpublished data
            # spc pre-public data is prepended by "spp", not "psp"
            # The psp_ files are located in the same directory after public release
            # but in that case users can just use the public (spdf) option. 
            prefix = 'spp_spc_'
            pathformat = 'sweap/spc/' + level + '/%Y/%m/spp_swp_spc_' + datatype + '_%Y%m%d_v0?.cdf'
    elif instrument == 'spe':
        prefix = 'psp_spe_'
        pathformat = 'sweap/spe/' + level + '/' + datatype + '/%Y/psp_swp_sp?_*_%Y%m%d_v??.cdf'
    elif instrument == 'spi':
        if username is None:
            prefix = 'psp_spi_'
            pathformat = 'sweap/spi/' + level + '/' + datatype + '/%Y/psp_swp_spi_*_%Y%m%d_v??.cdf'
        else:
            # unpublished data
            prefix = 'psp_spi_'
            pathformat = 'sweap/spi/' + level + '/' + datatype + '/%Y/%m/psp_swp_' + datatype + '*_%Y%m%d_v0?.cdf'
    elif instrument == 'epihi':
        prefix = 'psp_epihi_'
        pathformat = 'isois/epihi/' + level + '/' + datatype + '/%Y/psp_isois-epihi_' + level + '*_%Y%m%d_v??.cdf'
    elif instrument == 'epilo':
        prefix = 'psp_epilo_'
        pathformat = 'isois/epilo/' + level + '/' + datatype + '/%Y/psp_isois-epilo_' + level + '*_%Y%m%d_v??.cdf'
    elif instrument == 'epi':
        prefix = 'psp_isois_'
        pathformat = 'isois/merged/' + level + '/' + datatype + '/%Y/psp_isois_' + level + '-' + datatype + '_%Y%m%d_v??.cdf'

    # find the full remote path names using the trange
    remote_names = dailynames(file_format=pathformat, trange=trange, res=file_resolution)

    out_files = []

    if username is None:
        files = download(remote_file=remote_names, remote_path=CONFIG['remote_data_dir'], 
                        local_path=CONFIG['local_data_dir'], no_download=no_update,last_version=last_version)
    else:
        if instrument == 'fields':
            try:
                print("Downloading unpublished FIELDS Data....")
                files = download(
                    remote_file=remote_names, remote_path=CONFIG['fields_remote_data_dir'], 
                    local_path=CONFIG['local_data_dir'], no_download=no_update,
                    username=username, password=password, basic_auth=True,last_version=last_version
                )
            except:
                files = download(remote_file=remote_names, remote_path=CONFIG['remote_data_dir'], 
                                local_path=CONFIG['local_data_dir'], no_download=no_update,last_version=last_version)
        elif instrument in ['spc','spi']:
            try:
                print("Downloading unpublished SWEAP Data....")
                files = download(
                    remote_file=remote_names, remote_path=CONFIG['sweap_remote_data_dir'], 
                    local_path=CONFIG['local_data_dir'], no_download=no_update,
                    username=username, password=password, basic_auth=True,last_version=last_version
                )
            except:
                files = download(remote_file=remote_names, remote_path=CONFIG['remote_data_dir'], 
                                local_path=CONFIG['local_data_dir'], no_download=no_update,last_version=last_version)
        

    if files is not None:
        for file in files:
            out_files.append(file)

    out_files = sorted(out_files)

    if downloadonly:
        return out_files

    # find the list of varnames for RFS data
    # these files have > 1500 variables, but
    # we only load ~50
    if 'rfs' in datatype.lower() and varformat is None and varnames == []:
        varnames = rfs_variables_to_load(out_files)
        # we'll need the support data for the quality flags
        get_support_data = True

    tvars = cdf_to_tplot(out_files, suffix=suffix, prefix=prefix, get_support_data=get_support_data, 
                        varformat=varformat, varnames=varnames, notplot=notplot)

    if notplot:
        return tvars

    if time_clip:
        for new_var in tvars:
            tclip(new_var, trange[0], trange[1], suffix='')

    return tvars

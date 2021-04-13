import multiprocessing, glob, shutil, os, datetime, subprocess, math

import geopandas as gpd
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import cv2
import exiftool
import rasterio
from GPSPhoto import gpsphoto
import scipy.ndimage as ndimage
from skimage.transform import resize

from ipywidgets import FloatProgress, Layout
from IPython.display import display

from micasense import imageset as imageset
from micasense import capture as capture
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils

############ general functions for processing micasense imagery ############

def process_panel_img(panelNames, panelCorners=None, useDLS = True):
    location = None
    utc_time = None
    panel_irradiance = None
    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
        if panelCorners:
            panelCap.set_panelCorners(panelCorners)
        location = panelCap.location()
        utc_time = panelCap.utc_time()
    else:
        panelCap = None

    if panelCap is not None:
        if panelCap.panel_albedo() is not None and not any(v is None for v in panelCap.panel_albedo()):
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            #panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67] #RedEdge band_index order
            panel_reflectance_by_band = [0.493, 0.493, 0.492, 0.489, 0.491]

        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)    
        img_type = "reflectance"
    else:
        if useDLS:
            img_type='reflectance'
        else:
            img_type = "radiance"
    return(panel_irradiance, img_type, location, utc_time)

def get_warp_matrix(img_capture, max_alignment_iterations = 50):
    # TODO does this need img_type???
    # TODO ensure this is good for the Altum
    ## Alignment settings
    match_index = 2 # Index of the band 
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = 0 # for images with RigRelatives, setting this to 0 or 1 may improve alignment
    print("Aligning images. Depending on settings this can take from a few seconds to many minutes")
    # Can potentially increase max_iterations for better results, but longer runtimes
    warp_matrices, alignment_pairs = imageutils.align_capture(img_capture,
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels)

    #print("Finished Aligning, warp matrices={}".format(warp_matrices))
    return(warp_matrices)


def decdeg2dms(dd):
    is_positive = dd >= 0
    dd = abs(dd)
    minutes,seconds = divmod(dd*3600,60)
    degrees,minutes = divmod(minutes,60)
    degrees = degrees if is_positive else -degrees
    return (degrees,minutes,seconds)


def write_exif_csv(img_set, outputPath):
    header = "SourceFile,\
    GPSDateStamp,GPSTimeStamp,\
    GPSLatitude,GpsLatitudeRef,\
    GPSLongitude,GPSLongitudeRef,\
    GPSAltitude,GPSAltitudeRef,\
    FocalLength,\
    XResolution,YResolution,ResolutionUnits,\
    GPSImgDirection,GPSPitch,GPSRoll\n"

    lines = [header]
    for capture in img_set.captures:
        #get lat,lon,alt,time
        outputFilename = capture.uuid+'.tif'
        fullOutputPath = os.path.join(outputPath, outputFilename)
        lat,lon,alt = capture.location()
        #write to csv in format:
        # IMG_0199_1.tif,"33 deg 32' 9.73"" N","111 deg 51' 1.41"" W",526 m Above Sea Level
        latdeg, latmin, latsec = decdeg2dms(lat)
        londeg, lonmin, lonsec = decdeg2dms(lon)
        latdir = 'North'
        if latdeg < 0:
            latdeg = -latdeg
            latdir = 'South'
        londir = 'East'
        if londeg < 0:
            londeg = -londeg
            londir = 'West'
        resolution = capture.images[0].focal_plane_resolution_px_per_mm
        
        yaw, pitch, roll = capture.dls_pose()
        yaw, pitch, roll = np.array([yaw, pitch, roll]) * 180/math.pi

        linestr = '"{}",'.format(fullOutputPath)
        linestr += capture.utc_time().strftime("%Y:%m:%d,%H:%M:%S,")
        linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},'.format(int(latdeg),int(latmin),latsec,latdir[0],latdir)
        linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},{:.1f} m Above Sea Level,Above Sea Level,'.format(int(londeg),int(lonmin),lonsec,londir[0],londir,alt)
        linestr += '{},'.format(capture.images[0].focal_length)
        linestr += '{},mm,'.format(resolution)
        linestr += '{},{},{}'.format(yaw, pitch, roll)
        linestr += '\n' # when writing in text mode, the write command will convert to os.linesep
        lines.append(linestr)

    fullCsvPath = os.path.join(outputPath,'log.csv')
    with open(fullCsvPath, 'w') as csvfile: #create CSV
        csvfile.writelines(lines)
        
    return(fullCsvPath)

def save_images(img_set, outputPath, thumbnailPath, panel_irradiance, warp_img_capture, generateThumbnails = True, overwrite=False):
    """
    This function does ther actual processing of running through each capture within an imageset
    and saving as a .tiff.
    
    TODO change this to save images with rasterio as proper rasters
        only really necessary if we're going to map as opposed to use imgs as samples
    """
    # TODO this isn't working, these image alignments are terrible...
    # actually on the Altum they are very good
    # you need a relatively flat image and it should be at the same altitude
    # see https://github.com/micasense/imageprocessing/blob/de96aad06fc32a35597a8264190f81cd35206383/Alignment.ipynb
    warp_matrices = get_warp_matrix(warp_img_capture)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if generateThumbnails and not os.path.exists(thumbnailPath):
        os.makedirs(thumbnailPath)

    # Save out geojson data so we can open the image capture locations in our GIS
    # with open(os.path.join(outputPath,'imageSet.json'),'w') as f:
    #     f.write(str(geojson_data))

    try:
        irradiance = panel_irradiance+[0]
    except NameError:
        irradiance = None
    except TypeError:
        irradiance = None

    start = datetime.datetime.now()
    for i,capture in enumerate(img_set.captures):
        outputFilename = capture.uuid+'.tif'
        thumbnailFilename = capture.uuid+'.jpg'
        fullOutputPath = os.path.join(outputPath, outputFilename)
        fullThumbnailPath= os.path.join(thumbnailPath, thumbnailFilename)
        if (not os.path.exists(fullOutputPath)) or overwrite:
            if(len(capture.images) == len(img_set.captures[0].images)):
                capture.compute_undistorted_reflectance(irradiance_list=irradiance,force_recompute=True)
                capture.create_aligned_capture(irradiance_list=irradiance, warp_matrices=warp_matrices)
                capture.save_capture_as_stack(fullOutputPath, sort_by_wavelength=True)
                if generateThumbnails:
                    capture.save_capture_as_rgb(fullThumbnailPath)
        capture.clear_image_data()
    end = datetime.datetime.now()

    print("Saving time: {}".format(end-start))
    print("Alignment+Saving rate: {:.2f} images per second".format(float(len(img_set.captures))/float((end-start).total_seconds())))
    return(True)

def write_img_exif(fullCsvPath, outputPath):  
    if os.environ.get('exiftoolpath') is not None:
        exiftool_cmd = os.path.normpath(os.environ.get('exiftoolpath'))
    else:
        exiftool_cmd = 'exiftool'

    cmd = '{} -csv="{}" -overwrite_original {}'.format(exiftool_cmd, fullCsvPath, outputPath)
    print(cmd)
    #subprocess.check_call(cmd)
    return(True)

def process_micasense_subset(img_dir, panelNames, warp_img_dir=None, overwrite=False, panelCorners=None):
    """
    Testing function that takes in an image directorie and saves out processed imagery
    
    Various visualization functions for deciding ideal processing parameters
    
    """
    # ideally this goes through and picks out all panels and then chooses the closest one
    # right now it just takes one
    panel_irradiance, img_type, location, utc_time = process_panel_img(panelNames,panelCorners=panelCorners)
    if panel_irradiance:
        print("Panel irradiance calculated.")
    else:
        print("Not using panel irradiance.")
        
    imgset = imageset.ImageSet.from_directory(img_dir)
    
    if warp_img_dir:
        warp_img_capture = imageset.ImageSet.from_directory(warp_img_dir).captures[0]
        print('used warp dir', warp_img_dir)
    else:
        warp_img_capture = imgset.captures[0]
        
    outputPath = os.path.join(img_dir,'stacks')
    thumbnailPath = os.path.join(img_dir, 'thumbnails')
    
    if save_images(imgset, outputPath, thumbnailPath, panel_irradiance, warp_img_capture, overwrite=overwrite) == True:
        print("Finished saving images.")
        fullCsvPath = write_exif_csv(imgset, outputPath)
        if write_img_exif(fullCsvPath, outputPath) == True:        
            print("Finished saving image metadata.")
            
    return(outputPath)

def process_micasense_images(sea_dir, sky_dir, panelNames, overwrite=False, panelCorners=None):
    """
    Primary function that takes in sea and sky image directories and saves out processed imagery
    
    Various visualization functions for deciding ideal processing parameters
    
    """
    # ideally this goes through and picks out all panels and then chooses the closest one
    # right now it just takes one
    panel_irradiance, img_type, location, utc_time = process_panel_img(panelNames,panelCorners=panelCorners)
    if panel_irradiance:
        print("Panel irradiance calculated.")
    else:
        print("Not using panel irradiance.")
    
    # also will need to integrate the DLS as an alternative
    
    seaimgset = imageset.ImageSet.from_directory(sea_dir)
    seaOutputPath = os.path.join(sea_dir,'..','seastacks')
    seathumbnailPath = os.path.join(sea_dir, '..', 'seathumbnails')
    
    skyimgset = imageset.ImageSet.from_directory(sky_dir)
    skyOutputPath = os.path.join(sky_dir,'..','skystacks')
    skythumbnailPath = os.path.join(sky_dir, '..', 'skythumbnails')
    
    if save_images(skyimgset, skyOutputPath, skythumbnailPath, panel_irradiance, overwrite=overwrite) == True:
        print("Finished saving sky images.")
        fullCsvPath = write_exif_csv(skyimgset, skyOutputPath)
        if write_img_exif(fullCsvPath, skyOutputPath) == True:        
            print("Finished saving sky image metadata.")
    
    if save_images(seaimgset, seaOutputPath, seathumbnailPath, panel_irradiance, overwrite=overwrite) == True:
        print("Finished saving sea images.")
        fullCsvPath = write_exif_csv(seaimgset, seaOutputPath)
        if write_img_exif(fullCsvPath, seaOutputPath) == True:
            print("Finished saving sea image metadata.")
            
    return(seaOutputPath, skyOutputPath)


############ general functions for ingesting processed imagery and metadata ############

def key_function(item_dictionary):
    '''Extract datetime string from given dictionary, and return the parsed datetime object'''
    datetime_string = item_dictionary['UTC-Time']
    return datetime.datetime.strptime(datetime_string, '%H:%M:%S')

def load_img_fn_and_meta(img_dir, count=10000, start=0):
    df = pd.read_csv(img_dir + '/log.csv')
    df['filename'] = df['SourceFile'].str.split('/').str[-1]
    df = df.set_index('filename')
    img_metadata = []
    for file in glob.glob(img_dir + "/*.tif"):
        md = gpsphoto.getGPSData(file)
        md['full_filename'] = file
        filename = file.split('/')[-1]
        md['filename'] = filename
        # this isn't correctly loaded into the exifdata so pulling it into my own md
        md['yaw']   = (df.loc[filename]['    GPSImgDirection'] + 360) % 360
        md['pitch'] = (df.loc[filename]['GPSPitch'] + 360) % 360
        md['roll']  = (df.loc[filename]['GPSRoll'] + 360) % 360

        img_metadata.append(md)

    
    # sort it by time now
    img_metadata.sort(key=key_function)
    # cut off if necessary
    img_metadata = img_metadata[start:start+count]
    return(img_metadata)

def load_images(img_list):
    all_imgs = []
    for im in img_list:
        with rasterio.open(im, 'r') as src:
            all_imgs.append(src.read())
    return(all_imgs)

def retrieve_imgs_and_metadata(img_dir, count=10000, start=0, altitude_cutoff = 0):
    img_metadata = load_img_fn_and_meta(img_dir, count=count, start=start)
    idxs = []
    for i, md in enumerate(img_metadata):
        if md['Altitude'] > altitude_cutoff:
            idxs.append(i)
    
    imgs = load_images([img_metadata[i]['full_filename'] for i in idxs])
    imgs = np.array(imgs) / 32768 # this corrects it back to reflectance
    # TODO confirm with Anna that this is appropriate
    img = imgs / math.pi # this corrects from reflectance to remote sensing reflectance
    print('Output shape is: ', imgs.shape)

    # give the metadata the index of each image it is connected to so that I can sort them later and still
    # pull out ids to visualize from imgs
    img_metadata = [img_metadata[i] for i in idxs]
    i = 0
    for md in img_metadata:
        md['id'] = i
        i += 1
    return(imgs, img_metadata)


############ processing imagery ############

# many of these fcns are no longer used since they were based on a simple darkness proportion

def brightest_tube_pix(img, percent=0.0001):
    brightest = []
    for band in range(0,5):
        flat_img = img[band].flatten()
        count = int(-1*len(flat_img)*percent)
        ind = np.argpartition(flat_img, count)[count:]
        brightest.append(np.mean(flat_img[ind]))
    print('brightest pixels used:', count*-1)
    return(brightest)

def calculate_spectra_from_darkest_px(imgs, lowest_percent=0.75, band=4, return_imgs=False, visualize=False, sky=False):
    list_of_spectra = []
    sorted_img_list = []
    dark_idxs = []
    
    for i in range(0,imgs.shape[0]):
        print(i)
        if np.mean(imgs[i,0]) > 5 and not sky: # why am I doing this? I assume for thermal
            print(np.mean(imgs[i,0]))
            print('hitting this filter for brightness')
            continue
        # choose the percent to sort
        spectra = []
        #print(imgs.shape)
        num_to_sort = int(imgs[i,band].size * lowest_percent)
        #print('sorting ', num_to_sort)

        # efficiently sort the array
        flat_array = imgs[i,band].flatten()
        flat_array[flat_array == 0 ] = 1 # everything already equal to zero set to 1 to be ignored
        flat_array[flat_array < 0.0001 ] = 1 # take out dark pixels to be ignored
        idx = np.argpartition(flat_array, num_to_sort)[:num_to_sort]
        
        img_sorted = []
        blue_spec = None
        for img_idx in range(0,5):
            sorted_band = imgs[i,img_idx].flatten()
            if return_imgs:
                # add the sorted band to
                img_sorted.append(sorted_band)
            # take the mean of the darkest pixels from each band to add to the spectra
            spectra.append(np.mean(sorted_band[idx]))
            if img_idx == 0:
                blue_spec = np.mean(sorted_band[idx])
        if return_imgs:
            sorted_img_list.append(np.array(img_sorted))
        #if blue_spec <0.02:
        list_of_spectra.append(spectra)
        dark_idxs.append(i)
        
        if visualize:
            plt.hist(flat_array[idx], density=False)
            plt.axvline(x=np.mean(flat_array[idx]), color='red')
            plt.axvline(x=np.median(flat_array[idx]), color='black')
            print(i, np.median(flat_array[idx]))
            
    if return_imgs:
        return(sorted_img_list, idx)
    else:
        return(list_of_spectra, dark_idxs)
    
def remove_bright_pix(im, lowest_percent=0.75, band=0):

    sorted_imgs, lowest_idx = calculate_spectra_from_darkest_px(np.array([im]), lowest_percent=lowest_percent, band=band, return_imgs=True)
    
    dark_pix = np.zeros(np.array(sorted_imgs[0]).shape)
    dark_pix[:,lowest_idx] = sorted_imgs[0][:,lowest_idx]
    
    # cut out the super dark pix
    dark_pix[dark_pix < 0.0005] = 0
    dark_pix[dark_pix > 1 ] = 0
    dark_pix[dark_pix == 0] = np.nan
    
    return(dark_pix)


def visualize_darkest_pixels(im, lowest_percent=0.5, band=0, max_clim=0.1, min_clim=0, only_img=False):
    if im.shape[0] == 6: # because these are altum images
        im_flat = im[:-1].reshape(5,-1)
    else:
        im_flat = im.reshape(5,-1)
    # sort to get the darkest x pixels
    
    # visualize all bands and an RGB composite
    

    band_names = ['blue', 'green', 'red', 'red edge', 'nir']
    colors = ['blue', 'green', 'red', 'maroon', 'grey']
    
    fig, ax = plt.subplots(1,5, figsize=(16,14))
    for i,a in enumerate(ax):
        ims = a.imshow(im[i], cmap='jet', interpolation='none', vmax=max_clim, vmin=min_clim)
        a.set_title(band_names[i])
        fig.colorbar(ims, ax=a, fraction=0.046, pad=0.04)
        a.set_xticks([])
        a.set_yticks([])
#     ims = ax[5].imshow(im[0]/im[1], cmap='jet', vmax=10, vmin=0)
#     ax[5].set_title('blue/green')
#     ax[5].set_xticks([])
#     ax[5].set_yticks([])
#     fig.colorbar(ims, ax=ax[5], fraction=0.046, pad=0.04)
    #plt.savefig('openoceanfull.png')
    fig.show()
        
    if not only_img:
        
        # visualize all bands with the darkest pixels removed
        dark_pix = remove_bright_pix(im, lowest_percent=lowest_percent, band=band)


        fig, ax = plt.subplots(1,5, figsize=(16,14))
        for i,a in enumerate(ax):
            ims = a.imshow(dark_pix[i].reshape(im.shape[1:3]), interpolation='none', cmap='jet', vmax=max_clim, vmin=min_clim)
            a.set_title(band_names[i])
            fig.colorbar(ims, ax=a, fraction=0.046, pad=0.04)
            a.set_xticks([])
            a.set_yticks([])
        #plt.savefig('openoceanfilter.png')
        fig.show()


        fig, ax = plt.subplots(figsize=(12,8))
        colors = ['blue', 'green', 'red', 'grey', 'black']
        for i in range(0,5):
            ax.hist(dark_pix[i].flatten(), density=True, bins=50, color=colors[i], alpha=0.5)

        for i in range(0,5):
            print(np.count_nonzero(~np.isnan(dark_pix[i])))
        ax.set_xlim(0,0.1)

        return(dark_pix.reshape(5,im.shape[1], im.shape[2]))
    else:
        return(None)

############ chla retrieval algorithms ############

def L2chlor_a(Rrs443, Rrs488, Rrs547, Rrs555, Rrs667):
    ''' Use weighted MODIS Aqua bands to calculate chlorophyll concentration
    using oc3m blended algorithm with CI (Hu et al. 2012) '''

    # TODO update this with the proper coefficients
    thresh = [0.15, 0.20]
    a0 = 0.1977
    a1 = -1.8117
    a2 = 1.9743
    a3 = 2.5635
    a4 = -0.7218

    ci1 = -0.4909
    ci2 = 191.6590
    
    if Rrs443 > Rrs488:
        Rrsblue = Rrs443
    else:
        Rrsblue = Rrs488

    log10chl = a0 + a1 * (np.log10(Rrsblue / Rrs547)) \
        + a2 * (np.log10(Rrsblue / Rrs547))**2 \
            + a3 * (np.log10(Rrsblue / Rrs547))**3 \
                + a4 * (np.log10(Rrsblue / Rrs547))**4

    oc3m = np.power(10, log10chl)

    CI = Rrs555 - ( Rrs443 + (555 - 443)/(667 - 443) * \
        (Rrs667 -Rrs443) )
        
    ChlCI = 10** (ci1 + ci2*CI)

    if ChlCI <= thresh[0]:
        chlor_a = ChlCI
    elif ChlCI > thresh[1]:
        chlor_a = oc3m
    else:
        chlor_a = oc3m * (ChlCI-thresh[0]) / (thresh[1]-thresh[0]) +\
            ChlCI * (thresh[1]-ChlCI) / (thresh[1]-thresh[0])

    return chlor_a

def convert_to_ocean_color_gdf(chla_list, spectra_list, img_metadata):
    chla_dates = []
    for im in img_metadata:
        date_time_str = im['Date'] + ' ' + im['UTC-Time']

        date_time_obj = datetime.strptime(date_time_str, '%m/%d/%Y %H:%M:%S')
        chla_dates.append(date_time_obj)
    lons = []
    lats = []
    alts = []
    for im in img_metadata:
        lons.append(im['Longitude'])
        lats.append(im['Latitude'])
        alts.append(im['Altitude'])
        
    chla_df = pd.DataFrame(
    {'chla': chla_list,
     'Latitude': lats,
     'Longitude': lons,
     'Altitude' : alts,
     'spectra' : spectra_list,
     'time' : chla_dates})

    chla_gdf = gpd.GeoDataFrame(
        chla_df, geometry=gpd.points_from_xy(chla_df.Longitude, chla_df.Latitude))
    
    return(chla_gdf)


def vec_chla_img(blue, green):
    # this is a more efficient chla algorithm vectorized for numpy arrays
    a0 = 0.1977
    a1 = -1.8117
    a2 = 1.9743
    a3 = 2.5635
    a4 = -0.7218

    log10chl = a0 + a1 * (np.log10(blue / green)) \
        + a2 * (np.log10(blue / green))**2 \
            + a3 * (np.log10(blue / green))**3 \
                + a4 * (np.log10(blue / green))**4

    oc3m = np.power(10, log10chl)
    return(oc3m)

def chla_img(sky_spectra, dark_pix, wind_speed = 5):
    dp_shape = dark_pix.shape
    sky_rad_correction = np.reshape(np.array(sky_spectra) * (0.0256 + 0.00039 * wind_speed + 0.000034 * wind_speed * wind_speed), (5,1))
    water_leaving = dark_pix.reshape(5,-1) - sky_rad_correction
    
#     chlas = []
#     for i in range(water_leaving.shape[-1]):
#         chlas.append(L2chlor_a(water_leaving[0,i],water_leaving[0,i],water_leaving[1,i],water_leaving[1,i],water_leaving[2,i]))
#     chlas = np.array(chlas)
#     return(chlas.reshape(dp_shape[1:3]))
    chla_vec = vec_chla_img(water_leaving[0], water_leaving[1])
    return(chla_vec.reshape(dp_shape[1:3]))
    
def visualize_chla_across_thresholds(im, sky_spectra):
    thresholds = np.arange(0.1,1,0.1)
    fig, ax = plt.subplots(len(thresholds),1, figsize=(12,40))
    for i,lowest_percent in enumerate(thresholds):
        dark_pix = remove_bright_pix(im, lowest_percent=lowest_percent, band=0)
        full_chla_img = chla_img(sky_spectra, dark_pix.reshape(im.shape), wind_speed = 5)
        
        ims = ax[i].imshow(full_chla_img, interpolation='nearest', cmap='jet', vmax=0.5)
        ax[i].set_title(lowest_percent)
        #current_cmap = matplotlib.cm.get_cmap()
        #current_cmap.set_bad(color='yellow')
        fig.colorbar(ims, ax=ax[i], fraction=0.046, pad=0.04)
    
############ sun glint and reflected skylight removal algorithms ############

def sun_glint_removal(sea_spectra, sky_spectra, wind_speed, method='ruddick2006'):
    # this is the old approach based on ruddick and Mobley
    sky_spectra = np.median(sky_spectra,axis=0)
    water_leaving_spectra = []
    # TODO will add in Zhang and other approaches
    print((0.0256 + 0.00039 * wind_speed + 0.000034 * wind_speed * wind_speed))
    if method == 'ruddick2006':
        for water_spec in sea_spectra:
            water_leaving = np.array(water_spec) - np.array(sky_spectra) * (0.0256 + 0.00039 * wind_speed + 0.000034 * wind_speed * wind_speed)
            #water_leaving = water_leaving - water_leaving[4]
            # TODO if red edge is greater than ~.1 then it is cloudy and don't need wind correction just use 0.0256
            water_leaving_spectra.append(water_leaving)
    return(water_leaving_spectra)

def calculate_rho(sea_imgs, sky_imgs, blocked_spec, visualize=True):
    # this is the current approach to calculating rho based on sunblocked spectra
    
    # sea and sky img arrays are shape [img count, bands, rows, cols]
    # TODO calculate blurred sea and blurred sky for multiple images
    # TODO this could also be a surface fit to the image
    
    # note the smoothing process is quite slow and takes a while if you have too many images
    Lt_smooth_imgs = []
    for i in range(sea_imgs.shape[0]):
        Lt_smooth = ndimage.gaussian_filter(sea_imgs[i], sigma=(0, 20, 20), order=0)
        Lt_smooth_imgs.append(Lt_smooth)
    print(np.array(Lt_smooth_imgs).shape)
    Lt_smooth = np.mean(np.array(Lt_smooth_imgs), axis=0)
    # get lt minus lw
    Lt_Lw = (Lt_smooth.T - blocked_spec).T

    Lsky_smooth_imgs = []
    for i in range(sky_imgs.shape[0]):
        # flip lsky because the lowest part of the sea img is reflecting off the highest part of the sky
        Lsky = sky_imgs[i,:,::-1,:] # this flips the rows
        # smooth it out TODO could fit a surface to this too
        Lsky_smooth = ndimage.gaussian_filter(Lsky, sigma=(0, 20, 20), order=0)
        Lsky_smooth_imgs.append(Lsky_smooth)
        
    Lsky_smooth = np.mean(np.array(Lsky_smooth_imgs), axis=0)
    
    # divide this by the smoothed lsky
    rho = Lt_Lw / Lsky_smooth
    
    if visualize:
        fig, ax = plt.subplots(4,5, figsize=(18,16))
        for i in range(5):
            im = ax[0,i].imshow(Lt_smooth[i],cmap='jet', vmin=0.001, vmax=.07)
            fig.colorbar(im, ax=ax[0,i], fraction=0.046, pad=0.04)
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[0,i].set_title('original Lt (smooth)')

            im = ax[1,i].imshow(Lt_Lw[i],cmap='jet', vmin=0.0, vmax=.04)
            fig.colorbar(im, ax=ax[1,i], fraction=0.046, pad=0.04)
            ax[1,i].set_xticks([])
            ax[1,i].set_yticks([])
            ax[1,i].set_title('lt - lw (spec)')

            im = ax[2,i].imshow(Lsky_smooth[i],cmap='jet', vmin=0.05, vmax=.4)
            fig.colorbar(im, ax=ax[2,i], fraction=0.046, pad=0.04)
            ax[2,i].set_xticks([])
            ax[2,i].set_yticks([])
            ax[2,i].set_title('lsky smooth (flip)')

            im = ax[3,i].imshow(rho[i],cmap='jet', vmin=0.0, vmax=.25)
            fig.colorbar(im, ax=ax[3,i], fraction=0.046, pad=0.04)
            ax[3,i].set_xticks([])
            ax[3,i].set_yticks([])
            ax[3,i].set_title('rho')
            fig.show()
        
    return(rho)


def apply_rho(sea_img, sky_img, rho, visualize=True):

    # flip lsky because the lowest part of the sea img is reflecting off the highest part of the sky
    lsky = sky_img[:,::-1,:] # this flips the rows
    # smooth it out TODO could fit a surface to this too
    lsky_smooth = ndimage.gaussian_filter(lsky, sigma=(0, 20, 20), order=0)
    
    # rho needs to be the same size and it isn't when from the RedEdge
    
    rho_resized = []
    for i in range(rho.shape[0]):
        rho_resized.append(resize(rho[i], (sea_img.shape[1], sea_img.shape[2]), anti_aliasing=True))
    rho_resized = np.array(rho_resized)
    
    lw_img = sea_img - rho_resized * lsky_smooth
    print(lw_img.shape)
    
    if visualize:
        fig, ax = plt.subplots(5,5, figsize=(18,16))
        for i in range(5):
            im = ax[0,i].imshow(sea_img[i],cmap='jet', vmin=0.001, vmax=.06)
            fig.colorbar(im, ax=ax[0,i], fraction=0.046, pad=0.04)
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[0,i].set_title('original Lt')


            im = ax[1,i].imshow(lsky_smooth[i],cmap='jet', vmin=0.05, vmax=.3)
            fig.colorbar(im, ax=ax[1,i], fraction=0.046, pad=0.04)
            ax[1,i].set_xticks([])
            ax[1,i].set_yticks([])
            ax[1,i].set_title('lsky smooth (flip)')

            im = ax[2,i].imshow(rho_resized[i],cmap='jet', vmin=0.0, vmax=.15)
            fig.colorbar(im, ax=ax[2,i], fraction=0.046, pad=0.04)
            ax[2,i].set_xticks([])
            ax[2,i].set_yticks([])
            ax[2,i].set_title('rho (resized)')
            
            im = ax[3,i].imshow(rho_resized[i] * lsky_smooth[i],cmap='jet', vmin=0.0, vmax=.05)
            fig.colorbar(im, ax=ax[3,i], fraction=0.046, pad=0.04)
            ax[3,i].set_xticks([])
            ax[3,i].set_yticks([])
            ax[3,i].set_title('rho_resized * lsky_smooth')

            im = ax[4,i].imshow(lw_img[i],cmap='jet', vmin=0.005, vmax=.05)
            fig.colorbar(im, ax=ax[4,i], fraction=0.046, pad=0.04)
            ax[4,i].set_xticks([])
            ax[4,i].set_yticks([])
            ax[4,i].set_title('full lw')
            fig.show()
    return(lw_img)
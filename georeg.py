import geopandas as gpd
import pandas as pd

import os, glob, math, datetime
import multiprocessing

import cv2
import numpy as np
import matplotlib.pyplot as plt

import exiftool

import rasterio
import micasense.image as image

import cameratransform as ct


# space location / scaling factor - half the image

def spacetotopdown(top_im, cam, image_size, scaling):
    x1 = top_im.shape[0]/2 + cam.spaceFromImage([0,0])[0] / scaling
    y1 = top_im.shape[1]/2 - cam.spaceFromImage([0,0])[1] / scaling
    
    x2 = top_im.shape[0]/2 + cam.spaceFromImage([image_size[0]-1,0])[0] / scaling
    y2 = top_im.shape[1]/2 - cam.spaceFromImage([image_size[0]-1,0])[1] / scaling
    
    x3 = top_im.shape[0]/2 + cam.spaceFromImage([image_size[0]-1,image_size[1]-1])[0] / scaling
    y3 = top_im.shape[1]/2 - cam.spaceFromImage([image_size[0]-1,image_size[1]-1])[1] / scaling
    
    x4 = top_im.shape[0]/2 + cam.spaceFromImage([0,image_size[1]-1])[0] / scaling
    y4 = top_im.shape[1]/2 - cam.spaceFromImage([0,image_size[1]-1])[1] / scaling
    return(np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]))
    #return([x1,x2,x3,x4],[y1, y2, y3,y4])

def georegister_drone_imgs(path_name, new_name, im, lat, lon, alt, yaw, roll, pitch, focal_length, 
                           sensor_size, image_size, scaling=0.2, top_im_bounds=[-120,120,-120,120], visualize=True):
    cam = ct.Camera(
        ct.RectilinearProjection(focallength_mm=focal_length,
                sensor=sensor_size,
                image=image_size),
            ct.SpatialOrientation(elevation_m=alt, # img.altitude,
                    tilt_deg=pitch,
                    roll_deg=roll,
                    heading_deg=yaw,
                    pos_x_m=0, pos_y_m=0)
    )

    # gps pts are lat lon
    cam.setGPSpos(lat, lon, alt)
    
    #im = plt.imread(path_name)
    # this value is approximate and based on altitude, FOV, and viewing geometry
    print(scaling)
    top_im = cam.getTopViewOfImage(im, top_im_bounds, scaling=scaling, do_plot=False)
    
    image_coords = spacetotopdown(top_im, cam, image_size, scaling)
    print('img coords: ',image_coords)
    
    if visualize:
        fig,ax = plt.subplots(figsize=(9,9))
        ax.imshow(top_im, interpolation='none', vmin=0.01, vmax=0.05, cmap='jet')
        # ax.set_xlabel("x position in m")
        # ax.set_ylabel("y position in m");
        ax.scatter(image_coords[:,0],image_coords[:,1])
        fig.show()
        
    
    coords = np.array([
        cam.gpsFromImage([0               , 0]), \
        cam.gpsFromImage([image_size[0]-1 , 0]), \
        cam.gpsFromImage([image_size[0]-1 , image_size[1]-1]), \
        cam.gpsFromImage([0               , image_size[1]-1])]
    )
    
    gcp1 = rasterio.control.GroundControlPoint(row=image_coords[0,0], col=image_coords[0,1], x=coords[0,1], y=coords[0,0], z=coords[0,2], id=None, info=None)
    gcp2 = rasterio.control.GroundControlPoint(row=image_coords[1,0], col=image_coords[1,1], x=coords[1,1], y=coords[1,0], z=coords[1,2], id=None, info=None)
    gcp3 = rasterio.control.GroundControlPoint(row=image_coords[2,0], col=image_coords[2,1], x=coords[2,1], y=coords[2,0], z=coords[2,2], id=None, info=None)
    gcp4 = rasterio.control.GroundControlPoint(row=image_coords[3,0], col=image_coords[3,1], x=coords[3,1], y=coords[3,0], z=coords[3,2], id=None, info=None)
    
    
    # Register GDAL format drivers and configuration options with a
    # context manager.
    with rasterio.Env():
        # open the original image to get some of the basic metadata
        with rasterio.open(path_name, 'r') as src:
            profile = src.profile
            print('initial profile')
            print(profile)

            src_crs = "EPSG:4326"  # This is the crs of the GCPs
            dst_crs = "EPSG:4326"

            tsfm = rasterio.transform.from_gcps([gcp1,gcp2,gcp3,gcp4])
            profile.update(
                dtype=rasterio.float32,
                transform = tsfm,
                crs=dst_crs,
                width=top_im.shape[0], # TODO unsure if this is correct order
                height=top_im.shape[1]
            )

            print('updated profile')
            print(profile)
            #new_fn = path_name.split('/')[-1].split('.')[0]+'_georeferenced.tif'
            with rasterio.open(new_name, 'w', **profile) as dst:
                # we then need to transpose it because it gets flipped compared to expected output
                # TODO could and shout probably convert this back to an int
                dst.write(top_im.T.astype(rasterio.float32), 1)
                print('written out as ', new_name)
                
    return(True)


def format_alta_logs(fp):
    alta_logs = pd.read_csv(fp)
    # the last few hundred lines don't follow csv rules and are just summary data
    alta_logs.drop(alta_logs.tail(250).index,inplace=True)
    # this may introduce a tiny bit of error but I do this so I have a unique index for matching time
    # TODO could instead resample to seconds and take a mean
    alta_logs['id'] = alta_logs.index
    alta_logs['dt'] = pd.to_datetime(alta_logs.Date.apply(str)+alta_logs['GPS Time']+'.'+alta_logs.id.apply(str).str.zfill(9).apply(str).str.slice(start=3,stop=9), format='%Y%m%d%H:%M:%S.%f')
    alta_logs = alta_logs.set_index('dt')
    alta_logs = alta_logs.sort_index()
    return(alta_logs)


# TODO change it to this
# alta_micasense_georef(img_info_path_name, data_path_name, alta_logs, scaling=0.1, top_im_size=120):
def alta_micasense_georef(img_info_path_name, new_name, img_data, alta_logs, scaling, top_im_bounds=[-120,-120,-120,120], visualize=True):
    # open up the micasense camera object
    img = image.Image(img_info_path_name)
    
    # define intrinsic camera parameters
    f = img.focal_length # returns focal length in mm
    # dividing the pixel size by the focal plane resolution in mm to get sensor size - px/(px/mm) leaves mm
    sensor_size = img.size()[0] / img.focal_plane_resolution_px_per_mm[0], img.size()[1] / img.focal_plane_resolution_px_per_mm[1]    # in mm
    image_size = img.size()    # in px    
    
    img_idx = alta_logs.index.get_loc(img.utc_time, method='nearest')
    lat = float(str(alta_logs.iloc[img_idx]['Latitude'])[:2] + '.' + str(alta_logs.iloc[img_idx]['Latitude'])[2:])
    # TODO this assumes it is always negative with :3. be better.
    lon = float(str(alta_logs.iloc[img_idx]['Longitude'])[:3] + '.' + str(alta_logs.iloc[img_idx]['Longitude'])[3:])
    alt = alta_logs.iloc[img_idx]['GPS Height']
    yaw = alta_logs.iloc[img_idx]['Yaw']
    roll = 0
    pitch = 40 # TODO could pull this off the gimbal if it logged it
    
    print(lat, lon, alt, yaw, roll, pitch)
    
    
    if georegister_drone_imgs(img_info_path_name, new_name, img_data, lat, lon, alt, yaw, roll, pitch, f, 
                           sensor_size, image_size, scaling=scaling, top_im_bounds=top_im_bounds, visualize=visualize):
        print('Complete!')
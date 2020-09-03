# ocean_color
Retrieving ocean color from drone-based sensors

Process:

1. Measure LT, Li, and Ed
2. Filter out bad angles, times with variable cloud cover, high wind, etc
3. Correct sensor measurements to radiances
4. Correct image distortion from angle off nadir to correct geospatial footprint
5. Filter out bright outliers (specular reflection, white caps, etc)
    * Potentially only use the darkest ~10% of pixels
6. Normalize all spectra to reference Ed (downwelling irradiance)
7. Remove surface glint to get Lw
    * Simple version is just Mobley 1999
    * Simpler version if Mobley 1999 plus Ruddick 2006 residual correction
    * Regardless needs to be pixel based
    * *Best option is spectral glint correction per pixel

Basic glint removal is:

`Lw(θ,φ,λ) = LT(θ,φ,λ) - ρ(θ,φ,θ0,W) * Li(θ’,φ,λ,)`



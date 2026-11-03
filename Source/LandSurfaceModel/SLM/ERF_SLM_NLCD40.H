// NLCD40 vegetation parameters from WRF v4.7.1 run/VEGPARM.TBL.
// Categories 18-20 are undefined in WRF and must not be used as land.
#ifndef ERF_SLM_NLCD40_H
#define ERF_SLM_NLCD40_H
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
namespace SLMNLCD40 {
enum Column { Shdfac, Nroot, Rs, Rgl, Hs, LaiMin, LaiMax, EmisMin, EmisMax, AlbMin, AlbMax, Z0Min, Z0Max, Ztop, NumColumns };
AMREX_GPU_HOST_DEVICE inline bool valid_category (int category) noexcept
{
    return category >= 0 && category <= 40 && category != 18 &&
           category != 19 && category != 20;
}
AMREX_GPU_HOST_DEVICE inline int analog (int category) noexcept
{
    constexpr int categories[19] = {15, 13, 13, 13, 13, 16, 4, 1, 5, 7,
                                    6, 10, 11, 10, 10, 12, 12, 11, 11};
    return categories[category - 22];
}
AMREX_GPU_HOST_DEVICE inline amrex::Real fraction_weight
    (amrex::Real fraction, amrex::Real minimum, amrex::Real maximum) noexcept
{
    if (maximum <= minimum) { return 0.5; }
    const amrex::Real weight = (fraction - minimum) / (maximum - minimum);
    return weight < 0 ? 0 : (weight > 1 ? 1 : weight);
}
inline constexpr amrex::Real values[40][NumColumns] = {
    {0.7, 4, 125, 30, 47.35, 5, 6.4, 0.95, 0.95, 0.12, 0.12, 0.5, 0.5, 17}, // 1: Evergreen Needleleaf Forest
    {0.95, 4, 150, 30, 41.69, 3.08, 6.48, 0.95, 0.95, 0.12, 0.12, 0.5, 0.5, 35}, // 2: Evergreen Broadleaf Forest
    {0.7, 4, 150, 30, 47.35, 1, 5.16, 0.93, 0.94, 0.14, 0.15, 0.5, 0.5, 14}, // 3: Deciduous Needleleaf Forest
    {0.8, 4, 100, 30, 54.53, 1.85, 3.31, 0.93, 0.93, 0.16, 0.17, 0.5, 0.5, 20}, // 4: Deciduous Broadleaf Forest
    {0.8, 4, 125, 30, 51.93, 2.8, 5.5, 0.93, 0.97, 0.17, 0.25, 0.2, 0.5, 18}, // 5: Mixed Forest
    {0.7, 3, 300, 100, 42, 0.5, 3.66, 0.93, 0.93, 0.25, 0.3, 0.01, 0.05, 0.5}, // 6: Closed Shrubland
    {0.7, 3, 170, 100, 39.18, 0.6, 2.6, 0.93, 0.95, 0.22, 0.3, 0.01, 0.06, 0.5}, // 7: Open Shrubland
    {0.5, 3, 70, 65, 54.53, 0.5, 3.66, 0.93, 0.93, 0.25, 0.3, 0.01, 0.05, 0}, // 8: Woody Savanna
    {0.5, 3, 70, 65, 54.53, 0.5, 3.66, 0.92, 0.92, 0.2, 0.2, 0.15, 0.15, 0.5}, // 9: Savanna
    {0.8, 3, 40, 100, 36.35, 0.52, 2.9, 0.92, 0.96, 0.19, 0.23, 0.1, 0.12, 0.5}, // 10: Grassland
    {0.6, 2, 100, 30, 51.93, 1.75, 5.72, 0.95, 0.95, 0.14, 0.14, 0.3, 0.3, 0.5}, // 11: Permanent Wetland
    {0.8, 3, 40, 100, 36.25, 1.5, 5.68, 0.92, 0.985, 0.15, 0.23, 0.05, 0.15, 0.5}, // 12: Cropland
    {0.1, 1, 200, 999, 999, 1, 1, 0.88, 0.88, 0.15, 0.15, 0.5, 0.5, 0}, // 13: Urban and Built-Up
    {0.8, 3, 40, 100, 36.25, 2.29, 4.29, 0.92, 0.98, 0.18, 0.23, 0.05, 0.14, 0.5}, // 14: Cropland / Natural Veg. Mosaic
    {0, 1, 999, 999, 999, 0.01, 0.01, 0.95, 0.95, 0.55, 0.7, 0.001, 0.001, 0}, // 15: Permanent Snow
    {0.01, 1, 999, 999, 999, 0.1, 0.75, 0.9, 0.9, 0.38, 0.38, 0.01, 0.01, 0.02}, // 16: Barren / Sparsely Vegetated
    {0, 0, 100, 30, 51.75, 0.01, 0.01, 0.98, 0.98, 0.08, 0.08, 0.0001, 0.0001, 0}, // 17: IGBP Water
    {0, 0, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 0}, // 18: Unclassified
    {0, 0, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 0}, // 19: Fill Value
    {0, 0, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 0}, // 20: Unclassified
    {0, 0, 100, 30, 51.75, 0.01, 0.01, 0.98, 0.98, 0.08, 0.08, 0.0001, 0.0001, 0}, // 21: Open Water
    {0, 1, 999, 999, 999, 0.01, 0.01, 0.95, 0.95, 0.55, 0.7, 0.001, 0.001, 0}, // 22: Perennial Ice/Snow
    {0.3, 1, 200, 999, 999, 1, 1, 0.88, 0.88, 0.2, 0.2, 0.5, 0.5, 0}, // 23: Developed Open Space
    {0.27, 1, 200, 999, 999, 1, 1, 0.88, 0.88, 0.2, 0.2, 0.7, 0.7, 0}, // 24: Developed Low Intensity
    {0.02, 1, 200, 999, 999, 1, 1, 0.88, 0.88, 0.2, 0.2, 1.5, 1.5, 0}, // 25: Developed Medium Intensity
    {0.11, 1, 200, 999, 999, 1, 1, 0.88, 0.88, 0.2, 0.2, 2, 2, 0}, // 26: Developed High Intensity
    {0.01, 1, 999, 999, 999, 0.1, 0.75, 0.9, 0.9, 0.38, 0.38, 0.01, 0.01, 0.02}, // 27: Barren Land
    {0.8, 4, 125, 30, 54.7, 1, 5.16, 0.93, 0.94, 0.14, 0.17, 0.5, 0.5, 20}, // 28: Deciduous Forest
    {0.95, 4, 140, 30, 44, 3.08, 6.48, 0.95, 0.95, 0.12, 0.12, 0.5, 0.5, 17}, // 29: Evergreen Forest
    {0.8, 4, 125, 30, 51.93, 2.8, 5.5, 0.93, 0.97, 0.17, 0.25, 0.2, 0.5, 18}, // 30: Mixed Forest
    {0.7, 3, 170, 100, 39.18, 1, 4, 0.93, 0.95, 0.16, 0.3, 0.01, 0.04, 0.5}, // 31: Dwarf Scrub
    {0.7, 3, 300, 100, 42, 0.5, 3.66, 0.93, 0.93, 0.22, 0.3, 0.01, 0.05, 0.5}, // 32: Shrub/Scrub
    {0.8, 3, 40, 100, 36.35, 0.52, 2.9, 0.92, 0.96, 0.19, 0.23, 0.1, 0.12, 0.5}, // 33: Grassland/Herbaceous
    {0.6, 2, 40, 100, 60, 1.5, 5.65, 0.95, 0.95, 0.14, 0.14, 0.2, 0.2, 0.5}, // 34: Sedge/Herbaceous
    {0.6, 2, 40, 100, 60, 1, 2, 0.95, 0.95, 0.31, 0.31, 0.01, 0.01, 0}, // 35: Lichens
    {0.6, 2, 40, 100, 60, 1, 2, 0.95, 0.95, 0.24, 0.24, 0.01, 0.01, 0}, // 36: Moss
    {0.8, 3, 40, 100, 36.25, 1.56, 5.68, 0.92, 0.985, 0.17, 0.23, 0.05, 0.15, 0.5}, // 37: Pasture/Hay
    {0.8, 3, 40, 100, 36.25, 1.56, 5.68, 0.93, 0.985, 0.2, 0.25, 0.02, 0.1, 0.5}, // 38: Cultivated Crops
    {0.6, 2, 100, 30, 51.93, 0.7, 3.5, 0.95, 0.95, 0.14, 0.14, 0.4, 0.4, 20}, // 39: Woody Wetland
    {0.6, 2, 40, 100, 60, 0.7, 3.5, 0.95, 0.95, 0.12, 0.12, 0.2, 0.2, 0.5}, // 40: Emergent Herbaceous Wetland
};
} // namespace SLMNLCD40
#endif

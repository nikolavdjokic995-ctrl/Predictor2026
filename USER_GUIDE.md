# Predictor2026 – User Guide

## 1. Inputs
Add the study area boundary, historical LULC rasters, and predictor rasters.

### Historical LULC periods
Define at least one TRAIN period. A VALIDATION period is recommended for model testing.

### Predictors
Predictors can include terrain and proximity variables such as DEM, aspect, slope, distance to river, distance to road, or other explanatory rasters.

## 2. Reclassification
Use consistent LULC class codes across all rasters. The plugin expects a harmonized class scheme before prediction.

## 3. Transitions
Define which class-to-class transitions are allowed. These transitions control both the scenario editor and the simulation logic.

## 4. Scenarios
The plugin supports:
- BAU
- Depopulation
- Urban expansion
- Custom scenarios

Scenario behavior is controlled through transition multipliers. Higher multipliers strengthen a transition relative to the baseline model.

## 5. Target years
Add one or more years for future projection.

## 6. Validation
If a VALIDATION period is provided, the plugin evaluates the baseline model using:
- confusion matrix
- Overall Accuracy (OA)
- Kappa
- macro precision
- macro recall

## 7. Run
Choose an output directory and run the model. The plugin will export all generated rasters and reports to that folder.

## 8. Main outputs explained
### Predicted raster
Projected future LULC map.

### Change raster
Shows where change occurred between the input and predicted state.

### Confidence raster
Indicates relative confidence in the spatial allocation.

### Confusion matrix
Shows counts of correctly and incorrectly classified cells during validation.

### Validation report
Summarizes OA, Kappa, precision, and recall.

### Feature importance
Lists the relative influence of predictors when Random Forest is available.

## 9. Good practice
- Use aligned rasters when possible.
- Start with a small, well-tested predictor set.
- Validate before projecting far into the future.
- Use scenarios for future exploration, not for historical validation.

## Contact
Nikola V. Djokic  
nikolavdjokic995@gmail.com

# Predictor2026 – QGIS Plugin for LULC Change Prediction

Predictor2026 is a QGIS plugin for Land Use / Land Cover (LULC) change prediction based on a transparent hybrid modeling workflow.

## Core modeling approach
The plugin combines:
- Markov chains for transition quantities and temporal dynamics
- Random Forest for suitability estimation when supported in the local Python environment
- Cellular Automata neighborhood effects for spatial allocation
- Scenario-based transition multipliers for controlled future simulations

## Main capabilities
- Multi-period TRAIN and VALIDATION workflow
- Editable transition definitions
- Scenario-based simulation (BAU, Depopulation, Urban expansion, and custom scenarios)
- Transition matrix preview
- Custom target years
- Automatic raster alignment when needed
- Baseline validation with confusion matrix, Overall Accuracy, Kappa, macro precision, and macro recall
- Change raster and confidence raster export
- Feature importance export when Random Forest is available
- Run log and structured output reports

## Typical workflow
1. Select project metadata and boundary.
2. Add historical LULC periods and assign TRAIN / VALIDATION roles.
3. Add predictor rasters.
4. Define allowed transitions.
5. Select and adjust scenarios.
6. Add one or more target years.
7. Validate configuration.
8. Run prediction and review outputs.

## Output files
Depending on the selected options and available dependencies, the plugin can export:
- predicted LULC raster(s)
- change raster(s)
- confidence raster(s)
- confusion matrix CSV
- validation report JSON
- feature importance JSON
- run report JSON
- run log TXT

## Installation
1. Open QGIS.
2. Go to **Plugins → Manage and Install Plugins**.
3. Choose **Install from ZIP**.
4. Select the Predictor2026 ZIP package.
5. Enable the plugin.

## Notes
- Validation is performed only for the baseline model, not per future scenario.
- Scenario strength is controlled explicitly through transition multipliers.
- Random Forest functionality depends on local availability of required Python packages.

## Contact
**Nikola V. Djokic**  
Email: **nikolavdjokic995@gmail.com**

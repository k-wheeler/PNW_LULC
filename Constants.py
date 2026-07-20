"""Shared constants used across the pipeline."""

DATA_DIR      = './Data'
MODEL_DIR     = './Model_Outputs'
FIA_DATA_DIR  = './Data/FIA'
CCDC_OUTPUTS_DIR  = './CCDC_Outputs'

# Biomass -> carbon: IPCC default aboveground carbon fraction. Report x0.50 as a sensitivity
# for the overwhelmingly-conifer AOI.
IPCC_CARBON_FRACTION = 0.47
IPCC_CARBON_FRACTION_HI = 0.50
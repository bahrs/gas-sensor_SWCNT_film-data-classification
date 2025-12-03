| Column            | Description                                                                                                            |
|-------------------|------------------------------------------------------------------------------------------------------------------------|
| *file_extension*  | .parquet (`compression = 'brotli'`)                                                                                    |
| Index             | Datetime index                                                                                                         |
| V                 | Voltage applied to the sample with Keithley 2601B (V)                                                                  |
| I                 | Current through the sample measured by Keithley 2601B (A)                                                              |
| MFC_target        | Desired concentration of the target gas (ppm)                                                                         |
| flow_target_error | Difference (in % from desired flow) between actual flow through the Bronkhorst EL-FLOW MFC for target gas and desired |
| flow_carrier_error| Difference (in % from desired flow) between actual flow through the Bronkhorst EL-FLOW MFC for carrier gas and desired|
| meas_cycle        | Number of the measurement cycle (0–9)                                                                                 |


**NOTE**: data with inexplicable outliers at the start / end of the measurement file was cut manually. That's why files are of different length, although expected length is 1200*402 lines.


# Excerpts from the [article](https://linkinghub.elsevier.com/retrieve/pii/S0925400524008463) explaining the data acquisition process

## Sensor fabrication
As prepared SWCNT film (90% transmittance at the 550 nm wavelength) was transferred onto a 24x30 mm2 polycrystalline Al2O3 sample holder with 4 vacuum sputtered gold strip electrodes by a dry transfer technique. Electrodes were connected to wires with conductive epoxy resin (CW2400, Circuit Works®, USA). The sample holder was 0.8 mm thick and had an 11x11 mm2 opening in the center. The free-standing part of the CNT film had dimensions of approx. 11x6 mm2 (see the scheme and image in Fig. S1)

![Fig_S1](S1.png)

**Figure S1**. *Sensor setup scheme. 1. gas vessels; 2. mass flow controllers (Bronkhorst); 3. sealed sensor chamber; 4. exhaust; 5. multimeter (Keithley 2601B); 6. PC for data recording.*


## Sensor measurements
Sample heating was implemented via Joule heating by applying triangular voltage pulses using the Keithley SourceMeter® 2601B unit (Tektronix, Inc., USA) at a starting/ending voltage of 5 V, a peak voltage of 25 V, and a voltage sweep rate of 1 V/s. The data-recording rate was equal to approx. 8 data points per second with a latency of 130±5 ms and a cycle duration approximately of 50 s, 402 data points per heating cycle. The current values were recorded as an analytical signal. The sequence of gas mixture purging was repeated 10 times for each individual analyte (Fig. 1a), thus accumulating 1200 temperature modulation cycles within measurements of the sensor response to each analyte. The power consumption of the sensor was calculated to be just below 200 mW. A calibration of temperature vs. power applied to a free-standing SWCNTs film was performed using a pyrometer (Kelvin compact 1200D, Euromix, Russia, ±2.5oC). Temperature (T) vs. power (W) calibration results and a calculated temperature profile are presented in Fig. 1b,c.

![Fig_1](1.png)

**Figure 1**. *Sensor measurement protocol and SWCNT film temperature profile. (a) Representative sequence of Voltage vs. time (V(t)) measurement cycles repeated 10 times for each analyte. Changes in analyte concentration are highlighted with colored rectangles for concentrations of 10, 15, and 25 ppm in dry air. (b) Temperature (T) vs. power (W) dependence calibration curve acquired by pyrometer for SWCNT film. Area shaded with light blue shows the 95% confidence interval (95% CI) of the fit. (c) Calculated profile of temperature for a single cycle.*

The selection of target analytes was guided by their assessed relevance as air pollutants. For this research, NO2, H2S, and acetone were identified as the key analytes of interest. Notably, NO2 stands out as a prevalent air pollutant, with an annual average concentration of 5.2 ppb as reported by the World Health Organization (WHO) and 21 ppb according to Russian and European regulatory standards. The European Commission sets the threshold limit value time weighted average (TLV-TWA, maximum average concentration for a permissible 8-hour long exposure to the analyte at the workplace for healthy adult workers wearing protective equipment) for NO2 at 500 ppb. Similarly, H2S is recognized as a significant industrial air pollutant, with an American Conference of Governmental Industrial Hygienists (ACGIH)-recommended TLV-TWA of 1 ppm. Acetone falls under the category of volatile organic compounds and is also considered an air pollutant with a TLV-TWA of 250 ppm.
The concentrations selected for these analytes were chosen based on preliminary experiments and established threshold limit values. To ensure a more accurate assessment of absolute sensitivity and selectivity, we opted to use consistent concentrations for all target analytes. 
The sensing device was enclosed within a sealed Teflon chamber with a volume of 20 cm3. Gas flow was regulated using a Bronkhorst® mass flow controller (MFC), maintaining a total flow rate of 100 sccm. H2S, NO2, and acetone were introduced as target gases in concentrations of 10, 15, and 25 parts per million (ppm) in dry air, designated as 0 ppm. A schematic representation of the sensor setup is provided in Figure S1. Gas vessels containing the target gases (calibration gas vessels, Linde Gas Rus, 100 ppm) and a compressed dry air line (purified using a PNEUDRI MIDIplus dryer, England; ISO8573-1:2010 class 2, with water vapor concentration < 0.12 ppm) were connected to the sealed sensor chamber. Gas mixtures were cyclically alternated between all 3 concentrations of target analyte in a way that ensured analyte desorption and an even distribution of data for each analyte concentration (Fig. 1a).

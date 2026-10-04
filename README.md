# Reliability- and Uncertainty-Driven Adaptive Client Selection for Federated Remote Patient Monitoring

This repository contains the MATLAB simulation code for the research work:

**"Reliability- and Uncertainty-Driven Adaptive Client Selection for Federated Remote Patient Monitoring (RUACS)"**

The proposed RUACS framework performs adaptive client selection in Federated Learning (FL) by jointly considering:

* Sensor reliability
* Data quality
* Prediction uncertainty
* Client diversity through score-based probabilistic selection

The framework is designed for Federated Learning-based Remote Patient Monitoring (RPM), where healthcare clients may have different sensor noise levels and faulty sensing devices.

---

## Overview

Remote Patient Monitoring systems generate physiological data from distributed healthcare clients. In practical environments, sensor measurements can be affected by noise, faulty sensors, and measurement variability.

RUACS addresses this problem by assigning each client a selection score based on sensor reliability, data quality, and model prediction uncertainty.

The RUACS score is defined as:

$$
S_i =
w_r R_i +
w_q Q_i -
w_u U_i
$$

where:

* \(R_i\) = sensor reliability of client \(i\)
* \(Q_i\) = data-quality score of client \(i\)
* \(U_i\) = prediction uncertainty of client \(i\)
* \(w_r\) = reliability weight
* \(w_q\) = data-quality weight
* \(w_u\) = uncertainty weight

In the simulation:

$$
w_r = 0.45,\qquad
w_u = 0.30,\qquad
w_q = 0.25
$$

Higher reliability and data quality increase the selection score, while higher prediction uncertainty decreases it.

---

## Simulation Configuration

The current simulation uses:

| Parameter                      |     Value |
| ------------------------------ | --------: |
| Number of healthcare clients   |        50 |
| Federated Learning rounds      |       100 |
| Clients selected per round     |        10 |
| Training samples/client        |       200 |
| Test samples/client            |       100 |
| Physiological features         |         5 |
| Fault probability              |       20% |
| Base sensor noise              | 0.02–0.25 |
| Additional faulty-sensor noise |      0.40 |
| Learning rate                  |      0.06 |
| Local training epochs          |         6 |
| Warm-up rounds                 |         5 |
| Monte-Carlo samples            |         8 |
| Reliability weight             |      0.45 |
| Uncertainty weight             |      0.30 |
| Data-quality weight            |      0.25 |
| Selection temperature          |      0.08 |
| Random seed                    |        42 |

---

## Physiological Features

The simulation models five physiological measurements:

1. Heart Rate (HR)
2. Blood Oxygen Saturation (SpO2)
3. Temperature
4. Respiration Rate
5. Blood Pressure (BP)

Each healthcare client is assigned a client-specific physiological baseline.

Abnormal observations are generated using moderate physiological shifts.

The training and test datasets contain exactly 30% abnormal observations, providing consistent class proportions across clients.

---

## Sensor Fault Model

A client can be assigned a faulty sensor with probability:

$$
P(\text{fault}) = 0.20
$$

Faulty sensors receive additional measurement noise:

$$
\text{additional noise}=0.40
$$

Therefore, faulty sensors produce substantially noisier physiological measurements than normal sensors.

The simulation uses feature-specific sensor noise for HR, SpO2, temperature, respiration rate, and BP.

---

## Sensor Reliability

Sensor reliability is directly derived from the sensor noise level:

$$
R_i = \exp(-2.2n_i)
$$

where \(n_i\) represents the client-specific noise level.

The resulting reliability value is constrained to:

$$
0 \leq R_i \leq 1
$$

Higher sensor noise therefore results in lower reliability.

---

## Data Quality

Data quality is modeled separately from reliability.

The basic quality component is:

$$
Q_i = \exp(-1.6n_i)
$$

with a small independent random variation.

The resulting value is constrained to:

$$
0 \leq Q_i \leq 1
$$

This allows data quality and reliability to be related while avoiding making them identical quantities.

---

## Prediction Uncertainty

RUACS estimates prediction uncertainty using Monte-Carlo sensor perturbations.

For each client, the current global model is evaluated using multiple perturbed versions of the client's measurements.

The perturbations are generated according to the client's actual sensor noise level.

The uncertainty combines:

* Predictive entropy
* Prediction variability under sensor perturbations

The final uncertainty is calculated as:

$$
U_i =
0.40H_i +
0.60V_i
$$

where:

* \(H_i\) = mean predictive entropy
* \(V_i\) = normalized prediction variability

Eight Monte-Carlo samples are used in the current simulation.

---

## Federated Learning Model

The local prediction model is binary logistic regression:

$$
P(y=1|x)=\sigma(x^Tw+b)
$$

where \(\sigma(\cdot)\) is the sigmoid function.

Each selected client performs local stochastic gradient descent for six epochs.

The learning rate is:

$$
\eta = 0.06
$$

After local training, the server aggregates the client models using standard FedAvg:

$$
w^{global}
=
\frac{1}{K}
\sum_{i=1}^{K}w_i
$$

where \(K=10\) is the number of selected clients per round.

---

## RUACS Client Selection

During the first five communication rounds, random warm-up selection is used because the initial global model has not yet learned useful prediction patterns.

After the warm-up period, clients are selected using score-based probabilistic sampling.

The selection probability favors clients with:

* Higher reliability
* Higher data quality
* Lower prediction uncertainty

The temperature parameter is:

$$
T=0.08
$$

A lower temperature gives stronger preference to clients with higher RUACS scores while still allowing client diversity.

---

## Random Baseline

The repository also contains an identical Federated Learning implementation using random client selection.

The random baseline uses:

* The same 50 clients
* The same datasets
* The same local logistic-regression model
* The same learning rate
* The same number of local epochs
* The same number of selected clients
* The same FedAvg aggregation
* The same test dataset

The primary difference is the client-selection mechanism.

This provides a direct comparison between RUACS and random client selection.

---

## Evaluation Metrics

The simulation evaluates:

### 1. Global Test Accuracy

Overall classification accuracy on the test data from all healthcare clients.

### 2. Balanced Accuracy

Balanced accuracy is calculated from sensitivity and specificity:

$$
BA =
\frac{Sensitivity+Specificity}{2}
$$

### 3. F1-score

The F1-score combines precision and sensitivity:

$$
F1 =
\frac{2 \times Precision \times Sensitivity}
{Precision+Sensitivity}
$$

### 4. Prediction Uncertainty

The average uncertainty of selected clients is compared with the uncertainty of all clients.

### 5. Reliability-Uncertainty Correlation

Pearson correlation is calculated between final sensor reliability and final prediction uncertainty.

### 6. Client Selection Frequency

The number of times each healthcare client is selected during the 100 FL rounds is recorded.

---

## Generated Figures

The MATLAB code generates six figures.

### Figure 1 — Sensor Reliability

Shows the reliability score of all 50 healthcare clients.

### Figure 2 — Sensor Reliability vs Prediction Uncertainty

Shows the relationship between sensor reliability and prediction uncertainty, including a linear trend.

### Figure 3 — RUACS vs Random Client Selection

Compares global test accuracy across the 100 FL rounds.

### Figure 4 — Prediction Uncertainty

Compares the uncertainty of RUACS-selected clients with the uncertainty of all healthcare clients.

### Figure 5 — Client Selection Frequency

Shows how frequently each healthcare client is selected and compares the frequencies with the expected average.

### Figure 6 — Reliable and Faulty Sensors

Shows the reliability values of normal and faulty healthcare sensors.

---

## Reproducibility

The simulation initializes MATLAB's random number generator using:

```matlab
rng(42);
```

This provides a fixed starting random state for reproducibility.

The simulation can therefore be rerun using the same code and parameter configuration.

---

## Requirements

The simulation requires:

* MATLAB
* Basic MATLAB numerical and plotting functionality

No external toolbox is required by the core implementation.

---

## How to Run

1. Download or clone this repository.
2. Open `RUACS_main.m` in MATLAB.
3. Run the script.
4. The simulation will:

   * Generate healthcare clients
   * Generate sensor characteristics
   * Generate training and test data
   * Train the RUACS federated model
   * Train the random-selection baseline
   * Calculate evaluation metrics
   * Display final results
   * Generate six figures

---

## Repository Contents

```text
RUACS-Federated-Remote-Patient-Monitoring/
│
├── README.md
├── RUACS_main.m
├── LICENSE
│
└── results/
    └── README.md
```

---

## Research Context

This implementation is intended to support research on:

* Federated Learning
* Remote Patient Monitoring
* Healthcare IoT / IoMT
* Adaptive Client Selection
* Sensor Reliability
* Prediction Uncertainty
* Data Quality
* Faulty Sensor Detection
* Privacy-Preserving Healthcare AI

---

## Citation

If you use this code in academic work, please cite the associated RUACS research paper.

**Paper title:**

"Reliability- and Uncertainty-Driven Adaptive Client Selection for Federated Remote Patient Monitoring"

Citation information will be added after publication.

---

## License

This project is released for research and academic purposes.

See the `LICENSE` file for the applicable license.

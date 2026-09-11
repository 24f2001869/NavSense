# Jupyter Notebooks Directory Guide

This directory contains interactive Jupyter notebooks for sequential exploratory data analysis, signal visualization, and prototype verification.

---

## Notebook Index

1. **`01_dataset_exploration.ipynb`**: Parsing and inspecting raw IO-VNBD synchronized trip files and sensor channels.
2. **`02_sensor_visualization.ipynb`**: Plotting multi-axis accelerometer, gyroscope, magnetometer, and reference speed profiles.
3. **`03_coordinate_frames.ipynb`**: Demonstrating coordinate transformations between sensor frame ($b$), vehicle frame ($v$), and navigation frame ($n$).
4. **`04_dead_reckoning.ipynb`**: Interactive pure inertial strapdown mechanization and visualization of quadratic drift.
5. **`05_ekf.ipynb`**: Step-by-step implementation of the 15-state Error-State Kalman Filter with synthetic GNSS outages.
6. **`06_ai_velocity.ipynb`**: Training and evaluating forward-speed estimation models.
7. **`07_ai_ekf.ipynb`**: Closed-loop fusion of neural velocity predictions with the 15-state ESKF.

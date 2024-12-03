# Geometrical_Optics-master

Welcome to the Geometrical Optics Code Repository! This repository serves as a snapshot of the powerful codebase used to generate the results presented in our recent article. If you're looking to understand, reproduce, or build upon the geometrical optics simulations that were an integral part of our research, you've come to the right place.

## Introduction
Optical phenomena play a vital role in a wide range of scientific and engineering fields, from physics and astronomy to telecommunications and medical imaging. This repository contains the Geometrical Optics code that powered the simulations of Dark Field X-ray Microscopy (DFXM) and analyses discussed in our article, which is available [here](link-to-article).

Our codebase is designed to provide a clear and accessible implementation of Geometrical Optics principles, making it easy for researchers and enthusiasts to explore and experiment with optical systems. It can be used to simulate the behavior of x-rays as they interact with lenses, mirrors, prisms, and samples, helping you gain a deeper understanding of the inner workings of your material.

## Getting Started

1. **Clone the Repository:** Start by cloning this repository to your local machine using Git. You can do this by running the following command:

   ```bash
   git clone https://github.com/borgi-s/Geometrical_Optics-master.git
2. **Install Dependencies:** Make sure you have all the necessary dependencies installed. Detailed instructions will be added in the repository's documentation at a later date.

3. **Explore and Experiment:** Dive into the code, explore different optical scenarios, and use it to conduct your experiments or simulations.

4. **Reciprocal Space:** Use the 'generate_res.py' script to generate your reciprocal space resolution. Define the angular space that will be probed. This will create some pkl files with information for the direct space part of the resolution function.

5. **Direct Space:** Use either 'forward_model.py' or 'init_forward.py' to start generating images. The standard geometrical parameters of ID06 at the European Synchrotron Radiation Facility, where the microscope is set up experimentally, are the default settings.

## Usage
The Geometrical Optics code is versatile and can be used for various purposes, including:

- Simulating the behavior of light rays in different optical systems.
- Analyzing the formation of images by lenses, mirrors, and other optical components.
- Investigating optical aberrations and their effects on image quality.
- Designing and optimizing optical systems for specific applications.
- We encourage you to explore the code and adapt it to your specific research or educational needs. If you find any issues or have suggestions for improvements, please feel free to contribute.

## Descriptions

Here are some short descriptions of what some of the files contain, and how they can be used.

- 'image_processor.py' contain different functions and scripts to save and load DFXM images, analyse images with various function and much more.

- 'functions.py' has all the functions used by the direct space scripts

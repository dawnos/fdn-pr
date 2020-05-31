
Code for `Adversarial Feature Disentanglement for Place Recognition Across Changing Appearance`.

# Dependencies
We develop our code under anaconda, with following depenencies (not listed fully):
 - pytorch 1.1.0
 - torchvision 0.3.0
 - cuda 9.0
 - tensorboardx 1.7
 - toolz 0.9
 - matplotlib 3.0.3
 - hdf5storage 0.1.15
 - numpy 1.16.4
 - scikit-learn 0.21.1
 - pyyaml 5.1
 - py-opencv 3.4.2
 - attrdict 2.0.1
 - latex? (remove it later)

# Run
## MNIST toy case:
`python main.py config/mnist.yaml`

## Place recognition:
`python main.py config/pr.yaml`

# Notice
The code is under reconstruction. Full scripts to reproduce the results will be provided after acceptance.



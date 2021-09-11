# thermal_barrierlife_prediction

This is the result of the 'One-Hot-Encoating' team (Karin Hrovatin, Julius Polz, Nikolay Garabedian, Jazib Hassan, Ke Li, Sabrina Richter) for The Thermal Barrier Life Prediction Challenge (https://data-challenges.fz-juelich.de/web/challenges/challenge-page/84/overview) in the scope of the Helmholtz Herbst Hackathon 2021. All our model versions do several predictions on random crops of the same input image and aggregate these predictions to the median value. Model extensions include working on Fourier-Transformed images, including mixup samples during training and using the magnification as covariate in the dense part of the model.

## links
[slides for project flowcharts](https://docs.google.com/presentation/d/1TUbPHSYw5zZWDONORb0P053ieW91_Pb8aI3VtWpJl-s/edit?usp=sharing)

[mixup for data augmentation](https://arxiv.org/abs/1710.09412)

[fourier feature networks](https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb#scrollTo=OcJUfBV0dCww)

[Predicting Effective Diffusivity of Porous Media from Images by Deep Learning](https://www.nature.com/articles/s41598-019-56309-x#Sec2)

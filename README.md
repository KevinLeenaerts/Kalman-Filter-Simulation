
# [Kalman Filter] Simulation
This project provides a simulation of a [Kalman filter] where sensory data regarding both position and velocity are combined to gain a better estimate of a robot's position. Specifically developed for and by [FTC]-team 16383: Frits Philips Robotics team.

## Introduction

Localisation for robots is often a difficult task, where one had best be sceptical about the readings provided by sensors. However, if computers are one thing not, it's sceptical. Therefore the question arises of how we can make a computer a little less willing to blindly believe it's sensory input.

## The Problem
Altough this project can be made to cover a wider range of problems, it has been specifically tailored for one case. This is the case of [FTC]-team 16383 where the goal is to provided a better localisation of the robot using a [Kalman filter] to both filter input data as well as to fuse sensors in order to eliminate drift. The sensors are the following:

| Sensor           | Measured Property | Accuracy  | Accumulates Drift |
| ---------------- |:----------------- | ----------- | --------------|
| Odometry Wheels  | Position |  High | Yes |
| Camera           | Position      |  Low | No |

As you can see, the sensors have opposite properties regarding accuracy and drift. Therefore, we are looking for a way to combine both sensors in such a way that keeps the property of one sensor and eliminates that of the other sensor.

## Solution
Let us start with the solution for the odometry wheels. Since these accumulate drift over time, the proposed solution is to eliminate time by differentiating its input.

>$x_m = {x_r+d*t}$$
>
>${\frac {dx}{dt}}  (x_m)dx={{\frac {dx}{dt}}(x_r+d*t)dx}$$
>
>$v_m = {v_r+d}$$
>
>Where:
>* $x$ is the location
>* $v$ is the velocity
>* The subscript $_m$ denotes the measured value
>* The subscript $_r$ denotes the real value

This tells us that by differentiating the measured position we can gain information about velocity without accumulating drift from the sensor. However, the drift will still accumulate in the filter. 

To combat tihs, let us fuse this sensory data with the data from a camera that utilises [Vuforia] to gain a position estimate based on the known position of an image target. This data will be parsed into our filter as positional data, where the data from the odometry wheels will be parsed as velocity data.

Utilising this solution, we have succesfully moved the accumulating drift from the sensor to our filter where the error can be corrected using input from the camera.

[FTC]: https://www.firstinspires.org/robotics/ftc
[Kalman Filter]: https://en.wikipedia.org/wiki/Kalman_filter
[Vuforia]: https://www.google.com/search?q=vuforia&oq=vuforia&aqs=chrome.0.69i59l2j35i39j0i512l2j69i60l3.1484j0j7&sourceid=chrome&ie=UTF-8

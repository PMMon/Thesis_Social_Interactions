# Generate Dataset comprising Human-Space Interactions

You can create datasets that also include human-space interactions amongst human-human interactions between pedestrians. For this, run the script: 

```
python main/GenPedSpaceScene.py
```

By default, this will create a dataset that simulates the ZARA1 scenario. The dataset comprises 100 frames with a constant number of 8 pedestrians moving in each scene. Each pedestrian enters the scenario at one randomly selected side and leaves it at another randomly selected side. The default values of the *repulsive potential between pedestrians* are `V0 = 2` and `sigma = 0.8686`. The default values of the *repulsive potential between pedestrians and obstacles* are `U0 = 2` and `r = 0.4343`. The script has a number of command-line flags that you can use to configure the dataset you want to generate: 

## General Configurations

* `--run_list`: Determines whether to run the generation for various values of V0 and sigma defined in two lists *V0_list* and *sigma_list*. Default is `False`.

* `--nr_scenes`: Specifies the number of scenes that the dataset comprise. Default is 100. 

* `--scenario`: Specifies the scenario that should be simulated. Choose either `zara1`, `zara2`, `univ`, `hotel`, `eth`. Default is `zara1`. 

* `--phase`: Specifies whether the dataset should be a training, validation or test set. Choose either `train`, `val` or `test`. Default is `train`.

* `--a`: Specifies the width of the square/rectangle. Default is 20 meters.

* `--b`: Specifies the length of the rectangle. Default is 30 meters. 

* `--threshold`: As pedestrians approach their destinations, it occurs that they *overshoot* their goals. This value defines the minimal offset when they have reached their goal. Default is 0.02 meters.

## Initialization

* `--nr_agents`: Specifies the number of agents in each scene. Default is 8.

* `--v_max`: Defines the maximum possible initial velocity of a pedestrian. Default is 1.2 m/s.

* `--v_min`: Defines the minimum possible initial velocity of a pedestrian. Default is 0.4 m/s.

## Animation of Dataset

It is possible to create an animation of the dataset generated. The following flags control these animations:

* `--show_animation`: Determines whether to animate the simulated dataset or not. Default is `False`. 

* `--show_potential`: Determines whether to visualize the repulsive potential between pedestrians in the animation of the dataset. Default is `False`.

* `--agent_radius`: Specifies the radius of the circle that represents a pedestrian in the animation. Default is 0.35 meters. 

* `--create_background`: Determines whether to show and save an image of the created background of the simulation. Default is `False`. 

* `--scaling`: Specifies scaling factor to map pixel of input image to coordinates of scene. Default is 0.05.

## Repulsive Potentials and Social Forces

The Social Force Model by D. Helbing and P. Molnár (1995) is used to generate a dataset. This model uses repulsive and attractive forces to describe the motion behavior of pedestrians. The following flags control these forces: 

* `--V0`: Specifies the magnitude of the repulsive potential between pedestrians. Default is 2. 

* `--sigma`: Specifies the range of the repulsive potential between pedestrians. Default is 1.303.

* `--U0`: Specifies the magnitude of the repulsive potential between pedestrians and obstacles. Default is 2.

* `--r`: Specifies the range of the repulsive potential between pedestrians and obstacles. Default is 0.4343.

* `--tau`: Specifies relaxation time. Default is 0.5 seconds. 

* `--beta`: Determines the ratio to which extent social forces are exerted parallel and orthogonal to the direction of movement of a pedestrian. Set 1 for only parallel contributions. Default is 1.

* `--delta_t`: Specifies the time step size. Default is 0.4 seconds.

* `--twophi`: Angle that specifies the field of view 2*phi. Default is 200. 

* `--c`: Specifies out-of-view factor. Default is 0.5. 

## Visdom Configurations

It is also possible to view animations/images created during the generation process via the `visdom module`. The following flags control this process: 

* `--visdom`: Determines whether to show animations/videos in visdom. Default is `False`.

* `--viz_port`: Specifies port number for visdom connection. Default is 8098. 

* `--viz_server`: Specifies server name for visdom connection. Default is ' '. 

* `--viz_env`: Specifies name of visdom environment. Default is 'Socialforce_PedSpaceScene'.
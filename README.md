# A simple Fast fourier transform for the Modular hackathon.

[Implementation Link](https://github.com/martinvuyk/hackathon-fft).

This is my first time programming anything on a GPU beyond just using Tensorflow. As many people have said, it was surprisingly easy. I also decided to do something I'm familiar with and is solved in 1  dimension to avoid the many headaches that arise with multidimensional tensors.

I went for the Fast Fourier Transform because I've run this algorithm by hand in university and it is in my original field of study. Also because while I was doing it I realized how parallelizable it is. Most of the operations get boiled down to an operation between a pair of complex numbers that again get stored in two complex numbers.

The butterfly diagram is this original algorithm:

![image](./Butterfly%208%20Input%20Example.jpg)

Most of the work was figuring out how to setup the compile time constant arrays and how to index into them in parallel.

The implementation itself is the simplest form by constraining to 1D signals and having the imput be forced to be made up of a power of two length. A higher level function could then add padding etc. as a future improvement.

Computation-speed-wise I don't think this algorithm will fare badly, I just didn't have time to setup proper benchmarking. I haven't looked at what SOTA algorithms do, this is what came to me over the course of today (yesterday was all puzzle solving to get up to speed).
# pysingfel
This is a new repo of pysingfel maintained by Haoyuan. It is not quite compatible with the original with Zhaoyou's implementation.

The documentation is not complete. There are also known bugs not fixed in this repo.
The author is preparing this oral qualification exam and will return and fix them after April 13th.

Please contact hyli16@stanford.edu, if you come across any problems.

# Note
1. Throughout this package, the unit of all energy variables is eV. The unit of length variables is meter.
2. The wavenumber, wavevector is defined without 2\pi. It means that if k is the wavenumber, then k= 1/wavelength.
3. The wavevector is along z direction. Of course you can rotate it to obtain the wavevector along the other directions. But I have not checked whether it works for the other directions or not. The problem is that I might have implemented it in such a way that it only works when its along z axis.

# Problems to solve 
1. beam.py
        - set_focus function. The implementation is weird from my point of view.
        - get_focus
        - get_focus area
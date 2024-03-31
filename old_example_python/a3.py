# -*- coding: utf-8 -*-
"""
Assignment 3
Semester 1, 2022
ENGG1001
"""


import math
import numpy as np
import matplotlib.pyplot as plt
#from numpy import ndarray
__author__ = "<Konstantin Belov>, <s4731842>"
__email__ = "<Your Student Email>"


def nose_shape(x: float, L: float, Rb: float, d: float) -> float:
    """
     Parameters:
         x (float): x-position along body
         L (float): length of power-law body
         Rb (float): base radius of power-law body
         d (float): exponent in power-law body
     Returns:
         (float): y-position on body corresponding to x
     """
    y_pos = Rb*((x/L)**(d))
    return y_pos


def plot_nose_shape(L: float, Rb: float, d: float, number_samples: int) \
        -> None:
    """
    Parameters:
        L (float): length of power-law body
        Rb (float): base radius of power-law body
        d (float): exponent in power-law body
        number_samples (int): number of sample points for constructing plot
    Returns:
        None
    Side-effect:
        Plots power-law bpdy as x-y plot to screen
    """
    x = np.linspace(0, L, number_samples)
    y = nose_shape(x, L, Rb, d)
    plt.plot(x, y, marker="o")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.title(f"Power-law body: L={L}, Rb={Rb}, d={d}")
    plt.show()
    return


def drag_coeff_panel(x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Compute local c_d on a panel with end points (x0, y0) and (x1, y1)

    Parameters:
        x0 (float): x-position at left end of panel
        y0 (float): y-position at left end of panel
        x1 (float): x-position at right end of panel
        y1 (float): y-position at right end of panel
    Returns:
        (float): local drag coefficient
    """
    cdp = 2*((math.sin(math.atan((y1-y0)/(x1-x0))))**3)

    return cdp


def drag_coeff(L: float, Rb: float, d: float, number_panels: int) -> float:
    """
    Compute drag coefficient for a power-law body.

     Parameters:
         L (float): length of power-law body
         Rb (float): base radius of power-law body
         d (float): exponent in power-law body
         number_panels (int): number of panels for use in C_D calculation
     Returns:
         (float): drag coefficient
    """
    x = np.linspace(0, L, number_panels+1)

    total_area = 0
    initial_cd = 0

    y = nose_shape(x, L, Rb, d)

    for i in range(number_panels):

        panel_length = math.sqrt(((x[i+1]-x[i])**2)+((y[i+1]-y[i]))**2)
        panel_area = panel_length*2*math.pi*((y[i+1]+y[i])/2)

        cdp = drag_coeff_panel(x[i], y[i], x[i+1], y[i+1])

        total_area += panel_area
        initial_cd += cdp*panel_area

    C_D = (1/total_area)*initial_cd

    return C_D


def print_table(L: float, Rb: float, number_panels: int,
                d_start: float, number_entries: int, step: float) -> None:
    """
    Print table of C_Ds for various choices of exponent d.

    Parameters:
         L (float): length of power-law body
         Rb (float): base radius of power-law body
         number_panels (int): number of panels in C_D calculation
         d_start (float): first exponent
         number_entries (int): number of entries displayed in table
         step (float): difference between each exponent d
    Returns:
        None
    Side-effect:
        Table is printed to screen
    """
    d = d_start

    print('*' * 29)
    print('*', f"{'d':^12}", "*", f"{'C_D':^14}", '*', sep='')
    print('*' * 29)

    i = 0
    while i < number_entries:

        C_D = drag_coeff(L, Rb, d, number_panels)
        print('*', f"{d:^12.2f}", "*", f"{C_D:^14.4f}", '*', sep='')
        d += step
        i += 1

    print('*' * 29)
    return


class PowerLawBody(object):
    """Produces values used in plots and tables."""

    def __init__(self, L: float, Rb: float, d: float,
                 number_panels: int) -> None:
        """
        Parameters:
            L (float): length of power-law body
            Rb (float): base radius of power-law body
            d (float): exponent in power-law body
            number_panels (int): number of panels for use in C_D calculation
        Returns:
            None
        """
        self._L = L
        self._Rb = Rb
        self._d = d
        self._number_panels = number_panels

    def shape(self, x: float) -> float:
        """
        Parameters:
            x (float): x-position along body
        Returns:
            (float): y-position on body corresponding to x
        """
        y_pos = self._Rb*((x/self._L)**(self._d))
        return y_pos

    def drag_coeff(self) -> float:
        """
        Computes C_D for the body

        Returns:
            (float): drag coefficient
        """
        self.c_d = drag_coeff(self._L, self._Rb, self._d, self._number_panels)
        return self.c_d

    def set_design_param(self, new_d: float) -> None:
        """
        Parameters:
            new_d (float): value of d to replace d parameter
        Return:
            None
        """
        self._d = new_d

    def __call__(self, d: float) -> float:
        """
        Parameters:
            d (float): exponent in power-law body
        Returns:
            (float): drag coefficient for value d
        """
        self._d = d
        return self.drag_coeff()


def plot_drag_coeff(nose_cap: PowerLawBody, d_start: float,
                    d_stop: float, number_samples: int) -> None:
    """
    Parameters:
        nose_cap (PowerLawBody): the power-law body of interest
        d_start (float): the start-point power-law exponent for plotting
        d_stop (float): the stop-point power-law exponent for plotting
        number_samples (int): the total number of sample points for use in
        plotting

    Returns:
        None
    Side-effect:
        Plot to screen C_D as a function of d
    """
    x_cord = np.linspace(d_start, d_stop, number_samples)
    y_cord = np.zeros_like(x_cord)

    for i in range(0, number_samples):
        nose_cap.set_design_param(x_cord[i])
        y_cord[i] = nose_cap.drag_coeff()

    plt.plot(x_cord, y_cord, marker='o')
    plt.suptitle("Drag coefficients for power-law bodies")
    plt.xlabel('exponent, d')
    plt.ylabel('drag coefficient, C_D')
    plt.show()
    return


def golden_section_search(f, a: float, b: float, tol: float) -> float:
    """
    Use Golden Section Search to find a minimum of f.

    Parameters:
        f : function in one variable, accepts float, returns float
        a (float) : left end for search interval
        b (float) : right end for search interval
        tol (float): tolerance for stopping search

    Returns:
        (float): minimum of function f
    """

    # Step 1.
    g = 0.618034
    dL = a + (1 - g)*(b - a)
    dR = a + g*(b - a)
    fL = f(dL)
    fR = f(dR)

    # Step 2.
    while (dR - dL) >= tol:
        if fL < fR:
            b = dR
            dR = dL
            fR = fL
            dL = a + (1 - g)*(b - a)
            fL = f(dL)
        else:
            a = dL
            dL = dR
            fL = fR
            dR = a + g*(b - a)
            fR = f(dR)

            # Step 3.
    return 0.5*(dL + dR)


def main() -> None:
    """Entry point to interaction"""
    print("Implement your solution and run this file")


if __name__ == "__main__":
    main()

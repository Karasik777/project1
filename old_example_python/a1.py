"""
Assignment 1
Semester 1, 2022
ENGG1001
"""

__author__ = "<Konstantin Belov>, <s4731842>"
__email__ = "<k.belov@uqconnect.edu.au>"

HELP = """
    'i' - prompt for the input parameters
    'p <increments> <step>' - print out a table of distances to lift-off for different drag coefficients
    'h' - help message
    'q' - quit
"""
      
def prompt_for_inputs():
    """ Function that asks the user for inputs."""
    """Parameters: (mass_p, thrust_e, ref_area, air_density,
       initial_v, lift_v, start_x, time,drag_coeff)"""
    """Returns: Values tuple, drag_coeff tuple, for further computations."""
    """
       Returns: values: tuple[float, ...], drag˙coeff: float
       The first tuple of floats should contain the following values:
       mass, force, ref˙area, density, init˙velocity, lift˙velocity,
       start˙position, time˙inc.
       """

    
    mass_p = float(input('Input mass of the plane (in kg): '))
    thrust_e = float(input('Input engine force (in N): '))
    ref_area = float(input('Input reference area (in m^2): '))
    air_density = float(input('Input air density (in kg/m^3): '))
    initial_v = float(input('Input initial velocity at start of runway (in m/s): '))
    lift_v = float(input('Input lift-off velocity (in m/s): '))
    start_x = float(input('Input position of the start of the runway (in m): '))
    time = float(input('Input time increment (in secs): '))
    drag_coeff = float(input('Input drag coefficient: '))
    values = (mass_p, thrust_e, ref_area, air_density,
              initial_v, lift_v, start_x, time,)
    drag_coefft = ((drag_coeff),)
    return (values, drag_coeff)

def compute_trajectory(values, drag_coeff):
    """Function that takes the input data and computes velocity and position."""
    """Parameter: values, drag_coeff."""
    """Returns: position, velocity."""
    """
        Parameters:
        values (tuple[float, ...]): mass, force, ref˙area, density,
        init˙velocity, lift˙velocity, start˙position, time˙inc
        drag˙coeff (float): The drag coefficient.
        Returns:
         (tuple[tuple[float, ...], tuple[float, ...]]): The first tuple
        contains the positions (in meters) along the runway for each
        successive increment of time. The second tuple contains the
        velocities at successive increments in time.
   """
   
    (mass_p, thrust_e, ref_area, air_density,
     initial_v, lift_v, start_x, time,) = values
    pos_i = start_x
    vel_i = initial_v
    time = time
    position = ()
    velocity = ()
    while vel_i < lift_v:
        
            acceleration = (1/((mass_p)))*((thrust_e) -
            (0.5)*(air_density)*((vel_i)**2)*(ref_area)*(drag_coeff))
            
            pos_i = pos_i + vel_i * time + 0.5 * acceleration * time**2
            
            vel_i = vel_i + acceleration*(time)
            
            velocity += ((round(vel_i,3)),)
            
            position += (round(pos_i,3),)
            
    return position, velocity
            
def print_table(values, drag_coeff, increments, step):
    """Function that calculates runway distance as drag increases."""
    """Parameters: values, drag_coeff, increments, step."""
    """Returns: table of drag_coeff and runway distance."""
    """
        Parameters:
        values (tuple[float, ...]): mass, force, ref˙area, density,
        init˙velocity, lift˙velocity, start˙position, time˙inc
        drag˙coeff (float): The drag coefficient.
        increments (int): The number of drag coefficients displayed.
        step (float): The difference between each drag coefficient.

         Returns:
         None
    """
    final_drag = (drag_coeff + step*increments)
    print('*'*38)
    print('* Drag coefficient * Runway distance *')
    print('*'*38)
    for x in range (increments):
        positions, velocities = compute_trajectory(values, drag_coeff)
        print('*', f'{drag_coeff:^18.3f}', '*',
        f'{positions[-1]:^17.3f}', '*', sep='')
        drag_coeff += step  
    print('*'*38)
    print('')
        
def main():
    """Funcation that coordinates other funcitons, user interface."""
    
    test_user_input = False
    while True:
        command = input('Please enter a command: ').split()
        if command[0] == 'i':
            values, drag_coeff = prompt_for_inputs()
            test_user_input = True
        elif command[0] == 'h':
            print(HELP)
        elif command[0] ==  'p':
                if test_user_input == False:
                    print('Please enter the parameters first.')
                else:
                    increments = int(command[1])
                    step = float(command[2])
                    print_table(values, drag_coeff, increments, step)
        elif command[0] == 'q':
            g = input('Are you sure (y/n): ')
            if g == 'y':
                break
        else:
            if test_user_input == False:
                print('Please enter the parameters first. ') 
if __name__ == "__main__":
    main()
        
    
    

"""
Assignment 2.
Semester 1, 2022
ENGG1001
"""

# NOTE: Do not import any other libraries!

import csv

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


# Replace these <strings> with your name, student number and email address.
__author__ = "<Konstantin Belov >, <s4731842>"
__email__ = "<s4731842@student.uq.edu.au>" 


def file_import(file_name):
    """Returns: line count, headers array, timestamps array,
data array. Parameters: file_name - name of the file to import.
Program for reading data stored."""


    # Use the CSV module to read the file provided as input
    with open(file_name) as csv_file:
        # using the csv reader to scan the document 
        raw_data_list = csv.reader(csv_file, delimiter=',')
        # creating empty lists 
        headers = []
        timestamps = []
        data = []
        # allocating the key lines that contain
        # header information and data.
        header_row = 1
        first_row = 4
        for index, row in enumerate(raw_data_list):
            # Extracting the headers out of a document.
            if index == header_row:
                headers = row[2:]
            # Extracting timestamps and data from the document. 
            elif index >= first_row:
                timestamp_string = row[0]
                timestamp_convert = np.datetime64(timestamp_string)
                timestamps.append(timestamp_string)
                data.append(row[2:])
        linecount = index - first_row + 1
        # converting to array
        timestamps= np.array(timestamps, dtype = 'datetime64')
        data = np.array(data, dtype = 'float64')
    return linecount, headers, timestamps, data
                
        
def plot_data(timestamps, headers, data, gauges):
    """Returns plot of the data.
Parameters: timestamps array, headers array, data array,
gauges list. """
    # assign the x - axis. 
    x = timestamps - timestamps[0]
    # convert the time from nanoseconds to secods 
    x = x / np.timedelta64(1, 's')
    #plotting the data.
    for index,gauge in enumerate(headers):
        if gauge in gauges: 
            # assign the y - axis.
            y = np.array(data[::,index])
            plt.plot(x,y, label = gauge)
            plt.suptitle(f'Event Start: {timestamps[0]}')
            plt.xlabel( "Time From Event Start (s)")
            plt.ylabel('Strain ('r'$\mu\epsilon$'')')
        else:
            continue
        
    # graph formatting.
    plt.legend()    
    plt.grid() 
    plt.show()
    return


def centre_average(data: ndarray, window: int) -> ndarray:
    """Returns average data array.
Parameters: data array, window as integer."""
    spread = (window - 1)//2
    #Creating an array to fill with data. 
    average_data = np.zeros_like(data)
    #Getting the number of columns. 
    col_nums = len(data[0]) 
    b=0
    
    while b < spread:
    #Making the first row zero. 
        average_data[:,b] = 0
    #Making the last row zero. 
        average_data[:,b] = 0
        b+=1
    #Column count. 
    c = 0
    
    while c < col_nums:
        #Loop through rows. 
        for i in range(spread, len(data) - spread):
            # Calculate moving average.
            average_data[i,c] = np.average(data[i - spread \
                                                : i + spread + 1,c])
        c+= 1
        
    return average_data


def plot_avg_data(timestamps, headers, data, average_data, gauges):
    """Returns plot of average data.
Parameters: timestamps array, headers array, data array,
average data array, gauges list."""
    x = timestamps - timestamps[0]
    x = x / np.timedelta64(1, 's')
    x1 = x 
    for index,gauge in enumerate(headers):
        if gauge in gauges:
            y = np.array(data[::,index])         
            y1 = np.array(average_data[::,index])
            plt.plot(x,y, label = gauge)
            plt.plot(x1,y1, color = 'black', label = (gauge + ' Average'),\
                     linestyle = 'dashed')
            plt.suptitle(f'Event Start: {timestamps[0]}')
            plt.xlabel("Time From Event Start (s)")
            plt.ylabel('Strain ('r'$\mu\epsilon$'')')
        else:
            continue
    #graph formatting.
    plt.legend()    
    plt.grid() 
    plt.show()
    return


def analyse_event(timestamps, headers, data):
    """Returns stats array, max_rev_gauge as array elemet,
transit_time array, average_speed array, direction array.
Program for computing Max, Min, reversal of gauges readings,
finding gauge with highest range of readings, trasit time of vehicels
crossing the bridge, average speed of every vehicle, direction
of vehicles.
Parameters: timestamps, headers, data. """
    #5.1
    stats = np.zeros((3, len(headers)))
    # Loop through rows 
    b = 0
    
    while b < len(headers):
        #Loop through columns. 
        for index,value in enumerate(headers): 
            stats[0,index] = np.max(data[:,index])
        b+=1
    d = 0                            
    while d < len(headers):
        for index, value in enumerate(headers):
            stats[1, index] = np.min(data[:,index])
        d+=1
    n = 0 
    while n < len(headers):
        for index, value in enumerate(headers):
            stats[2, index] = stats[0,index] - stats[1, index]
        n +=1
           
    #5.2
    #Max revers gauge. 
    max_rev = np.where(stats == np.max(stats[2,:]))  
    max_rev_array = max_rev[1]
    max_rev_column = max_rev_array[0]
    max_rev_gauge = headers[max_rev_column]
    
    #5.3
    # finding the transit time of the vehicle. 
    VW_8_pos = headers.index("VW_8")
    VW_28_pos = headers.index("VW_28")

    VW_8_max_tuple = np.where(data == np.max(data[:,VW_8_pos]))
    VW_8_max_array = VW_8_max_tuple[0]
    VW_8_max_row  = VW_8_max_array[0] 
    
    VW_28_max_tuple = np.where(data == np.max(data[:,VW_28_pos]))
    VW_28_max_array = VW_28_max_tuple[0]
    VW_28_max_row = VW_28_max_array[0]
    #Times requried for calculation.  
    t1 = timestamps[VW_8_max_row]
    t2 = timestamps[VW_28_max_row]

    transit_time = (((t2-t1)/ np.timedelta64(1, 's'))).astype(np.float64)
    
    #5.4
    # distance between VW_8 and VW_28 78.52m.
    average_velocity = (78.52 / transit_time)
    average_velocity = average_velocity * 3.6
    
    if average_velocity < 0:
        average_speed = average_velocity* -1
    else:
        average_speed = average_velocity
    #5.5
    if transit_time < 0:
        direction = 'NE'
    else:
        direction = 'SW'
    return stats, max_rev_gauge, transit_time, average_speed, direction

        
def analyse_all(file_dir, file_list):
    """Returns event_start_time, average_speed_array,
direction_array, strain_reverse_array, average_data_bool.
Program reads every file, gathering data for analysis. 
Parameters: file directory, file list. """
    #6.1
    #Creating necessary arrays.  
    event_start_time  = np.zeros(len(file_list), 'datetime64[us]')
    average_speed_array = np.zeros(len(file_list), float)
    direction_array = np.zeros(len(file_list), '<U2')
    strain_reverse_array = np.zeros(len(file_list), float)
     
    gauge_position = input('Enter the gauge to analyse: ')
    data_used = input('Do you wish to smooth the data (y/[n]): ')

    average_data_bool = ''
    
    if data_used == 'y':
        average_data_bool = True
        window = input('Enter the smoothing window width: ')
        window = int(window)
        
    elif data_used == 'n':
        average_data_bool = False
        
    #Reading all files. 
    for file_name in file_list: 
        linecount, headers, timestamps, data = file_import(f'data/{file_name}')
         
        if average_data_bool == True:
            average_data = centre_average(data, window)
            data = average_data

        stats, max_rev_gauge, transit_time, average_speed,\
               direction = analyse_event(timestamps, headers, data)
        #6.1 
        event_start_time[file_list.index(file_name)] = timestamps[0]
        #6.2
        average_speed_array[file_list.index(file_name)] = average_speed
        #6.3
        direction_array[file_list.index(file_name)] = direction
        #6.4
        strain_reverse_array[file_list.\
                             index(file_name)] = stats[2,headers.\
                                                       index(gauge_position)]
    
    return event_start_time, average_speed_array, \
           direction_array, strain_reverse_array, average_data_bool


def main() -> None:
    """Returns: none.
Human interface program
for running multiple functions
as a single program with algorithm
to suit user. 
Parameters: none. """
        
    file_list = ['Event.20210501_000900.dat',
'Event.20210501_040539.dat',
'Event.20210501_042843.dat',
'Event.20210501_043032.dat',
'Event.20210501_043355.dat',
'Event.20210501_044530.dat',
'Event.20210501_044936.dat',
'Event.20210501_045345.dat',
'Event.20210501_050205.dat',
'Event.20210501_061505.dat',
'Event.20210501_073827.dat',
'Event.20210501_080041.dat',
'Event.20210501_080634.dat',
'Event.20210501_082714.dat',
'Event.20210501_083655.dat',
'Event.20210501_084749.dat',
'Event.20210501_085128.dat',
'Event.20210501_093107.dat',
'Event.20210501_102148.dat',
'Event.20210501_111906.dat',
'Event.20210501_115728.dat',
'Event.20210501_131551.dat',
'Event.20210501_134132.dat',
'Event.20210501_142211.dat',
'Event.20210501_144511.dat',
'Event.20210501_192032.dat',]
    
    test_user_input = False
    average_data_bool1 = ''
    while True:
        command = list((input('Please enter a command: ')).split(' '))
        average_data_bool1 = False
        if command[0] == 'h':
            print(
"""
    'h' - Help message
    'l' - Load a given event file for analysis
    's' - Smooth the raw data
    'p <gauges>' - Plot the data for strain gauges (separated by spaces)
    'a' - Analyse all event files
    'q' - Quit the program
""")
            
        elif command[0] == 'l':
            file_name = ''
            while file_name not in file_list:
                file_name = input('Enter the name of the event file: ')
                if file_name not in file_list:
                    print('This file does not exist.')
                    
            if file_name in file_list:
                test_user_input = True
                if test_user_input == True:
                    linecount, headers, timestamps, \
                               data = file_import('data/'+file_name)
                    print(f"{linecount} lines of data "
                          f"read from {file_name}.")
                    
                    stats, max_rev_gauge, transit_time, average_speed,\
                           direction = analyse_event(timestamps, headers, data)
                    print(f"{max_rev_gauge} measured the greatest "
                          f"strain reversal in {file_name}")
                    
                    test_user_input = True
            else:
                test_user_input = False
                print('This file does not exist.')

                                               
        elif command[0] == 'p':
            gauges = []
            gauges = command[1:]
            if test_user_input == False and len(gauges) >= 0:
                while len(gauges) >= 0:
                        print('You must load a file before proceeding...')
                        
                        break
            elif test_user_input == True and len(gauges) == 0:
                    print('No gauges selected to plot.')
                    
            else:
                if average_data_bool1 == False:
                    gauges = command[1:]
                    print('Creating plots...')
                    
                    plot_data(timestamps, headers, data, gauges)
                    
                if average_data_bool1 ==True:
                    gauges = command[1:]
                    print('Creating plots...')
                    
                    plot_avg_data(timestamps, headers,\
                                  data, average_data, gauges)

                
                
        elif command[0] == 's':
            if test_user_input == False:
                print('You must load a file before proceeding...')
            else:
                window = int(input('Enter the smoothing window width: '))
                
                average_data = centre_average(data, window)
                stats, max_rev_gauge, transit_time, \
                       average_speed, direction = analyse_event\
                       (timestamps, headers, data)
            
                max_range_gauge = np.max(stats[2,:])
                index_1 = np.where(stats == max_range_gauge)
                index_2 = index_1[1]
            
                maximum_reading = stats[0,index_2[0]]
                maximum_average_reading = np.max(average_data[:,index_2[0]])

                smoothing_index = ((np.max(stats[0,:] /
                                           np.max(average_data[:,index_2])))\
                                   * 100) - 100

                print(f"The maximum strain measured at {max_rev_gauge} has"
                      f"been reduced by {round(smoothing_index,2)}%.")

                average_data_bool1 = True
                
        elif command[0] == 'a':

            event_start_time, average_speed_array, \
                              direction_array,strain_reverse_array, \
                              average_data_bool  \
                              = analyse_all('data/', file_list)
            print('Creating plots...')
            speed_limit_imposed = \
                                float(input('Enter a speed limit to impose: '))

            vehicles_speeding = 0 
            for i in average_speed_array:
                if i > speed_limit_imposed:
                    vehicles_speeding+=1
                    
            if average_data_bool == True:
                print(f"{vehicles_speeding} vehicles detected exceeding the "
                      f"imposed speed limit of {speed_limit_imposed}km/hr"
                      f"using averaged data.")
            

            elif average_data_bool == False:
                print(f"{vehicles_speeding} vehicles detected exceeding "
                      f"the imposed speed limit of {speed_limit_imposed}"
                      f"km/hr using raw data.")
                
        elif command[0] == 'q':
            confirmation = input('Are you sure? (y/[n]): ')
            if confirmation == 'y':
                break

if __name__ == "__main__":
    main()

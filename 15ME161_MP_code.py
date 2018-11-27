import matplotlib.pyplot as plt
import math as m
import numpy as np
#-------------------------------------------------------------------------------------
# Atmospheric refraction using Newton's Corpuscular Theory
T = [1*m.pi/180]                                                       # Observation angle with respect to earth's normal
r0 = 6366                                                              # radius of the earth
dr = 0.2                                                               # incremental distance in the atmosphere
r = r0                                                                 # initialising radius
K = 0.003161                                                          # Constant calculated by Newton for Observation angle = 0
H = 10                                                                 # Max. distance of atmosphere (km)
i = 0
Coord = [0,0]
while (r-r0) <= H:                                                     # Checking if it has reached the max. height
    T.append(T[i] + dr*(-m.tan(T[i]))*(1/r + ((r0 - r)/H)*m.exp((r0 - r)/H)/(1+K*m.exp((r0 - r)/H))))
    Coord = [Coord[0]+dr*m.sin(-T[i]),Coord[1]+dr*m.cos(T[i])]         # Incrementing the co-ordinates
    plt.scatter(Coord[0],Coord[1])
    plt.xlabel('Distance along earth surface(km)')
    plt.ylabel('Atmospheric height(km)')
    plt.title('Refraction of light in atmosphere using Corpuscular Theory')
    r += dr
    i += 1
plt.pause(1)
plt.clf()
#-----------------------------------------------------------------------------------

xplot = np.linspace(-5, 5, 10)
yplot = np.linspace(-5, 5, 10)
dl = 0.5                                                              # Increment of distance for each step

plt.ion()

def p1plot(x):                                                         # Plot a circle
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(x)
def p2plot(x,y):                                                       # Plot 2 circles
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(x)
    ax.add_artist(y)
def p3plot(x,y,z,a,b):                                                 # Plot 3 points and 2 circles
    plt.scatter(x[0],x[1])
    plt.scatter(y[0],y[1])
    plt.scatter(z[0],z[1])
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(a)
    ax.add_artist(b)
#-----------------------------------------------------------------------------------------------------
def line(x,M,r,x1,y1):                                                 # Gives the y co-ordinate of the tangent to the circles
    return M*(x-x1) - r*m.sqrt(1+M**2) + y1
def circle1(x,r,x1,y1):                                                # Gives the y co-ordinate of the circle1 for a given x
    return y1 - m.sqrt(r**2 - (x-x1)**2)
def circle2(x,r,x2,y2):                                                # Gives the y co-ordinate of the circle2 for a given x
    return y2 - m.sqrt((r/2)**2 - (x-x2)**2)
#-----------------------------------------------------------------------------------------------------
def f_1(x,M,r,x1,y1):                                                  # Function whose root is tangent point to circle1
    return circle1(x,r,x1,y1) - line(x,M,r,x1,y1)
def derivfn_1(x,M,r,x1,y1):                                            # Derivative the above function
    return -(x - x1)/circle1(x,r,x1,y1) - M
def derivfn_11(x,r,x1,y1):                                             # Double derivative of the function
    return (-circle1(x,r,x1,y1) - (x-x1)**2/circle1(x,r,x1,y1))/circle1(x,r,x1,y1)**2

def f_2(x,M,r,x2,y2):                                                  # Function whose root is tangent point to circle2
    return circle2(x,r,x2,y1) - line(x,M,r,x2,y2)
def derivfn_2(x,M,r,x2,y2):                                            # Derivative the above function
    return -(x - x2)/circle2(x,r,x2,y2) - M
def derivfn_22(x,r,x2,y2):                                             # Double derivative of the function
    return (-circle2(x,r,x2,y2) - (x-x2)**2/circle2(x,r,x2,y2))/circle2(x,r,x2,y2)**2
#-----------------------------------------------------------------------------------------------------
def newton_raphson(f,derivfn,M,r,x1,y1,initial_guess,tol):             # Gives Newton Raphson solutions
    x = initial_guess
    error = 9
    i = 0
    while error > tol:
        error = abs(f(x, M, r, x1, y1)/derivfn(x, r, x1, y1))
        x = x - error
        i = i+1
    return [x, line(x, M, r, x1, y1)]
#-----------------------------------------------------------------------------------------------------
def refraction(n1,n2,Radii,width,Pos,Refract):                         # Main function which runs the simulation
    '''
    The function plots the trajectory of 3 collinear points representing a light beam.
    The point where particle 1 (left most particle) meets the new medium is called as the origin (x1,y1)
    The remaining 2 particles meet the new medium at (x2,y2) and (x3,y3) respectively.
    Once the particles reach the interface of media, they emanate wavelets (circles) whose radius is increasing
    at (1/n)th speed in the previous medium, where n = ratio of refractive indices. Tangent is drawn to the
    2 wavelets from the particle 3 when it reaches the interface. The light propagates in the direction perpendicular
    to the common tangent to all the wavelets.
    :param n1: refractive index of medium 1
    :param n2: refractive index of medium 2
    :param Radii: Array of distance values particle 1 has travelled in previous media
    :param width: Width of the beam (thickness)
    :param Pos: Array consisting of the origins in each refraction function call
    :param Refract: Array of refracted angle values in previously travelled media
    :return: returns [Radii, width, Pos, Refract] to be used as inputs for the next function call
    '''
    theta = Refract[len(Refract)-1]                                    # Assign the previous refraction angle as the incident angle
    P = []                                                             # Array used for plotting inside this function
    n = n2/n1                                                          # Ratio of refractive indices
    refract_angle = m.pi*10                                            # Declaring refraction angle
    [x1,y1] = [0,0]                                                             # Declaring origin (point where particle 1 meets next medium
    for i in range(len(Radii)):
        x1 = x1 + Radii[i]*m.sin(Refract[i])
        y1 = y1 - Radii[i]*m.cos(Refract[i])                           # contact point of particle 1 with medium 2
    [R1,R2,R3] = [0,0,0]
    x2 = x1 + width / (2 * m.cos(theta))                               # contact point of particle 2 with medium 2
    y2 = y1
    x3 = x1 + width / m.cos(theta)                                     # contact point of particle 3 with medium 2
    y3 = y1
    M = width*m.sin(theta)/m.sqrt((2*n)**2 - (width*m.sin(theta)**2))  # slope of the tangent to the circle
    r = width*m.tan(theta)/n                                           # final radius of the wavelet before tangent is drawn
    Pos.append([[x1,y1],[x2,y2],[x3,y3]])                              # Appending the new origin to Pos array
    P = Pos[:]                                                         # P is the local 3D array used for plotting
    P.append([[x1,y1],[x1+width/2*m.cos(theta),y1+width/2*m.sin(theta)],[x1+width*m.cos(theta),y1+width*m.sin(theta)]]) #Appending the initial position to P
    P = np.array(P)
    X = int(len(P))                                                    # Computing the length of the 3d array P
    while R1* m.cos(refract_angle)<= 4:
        plt.plot([P[X-1,:,0]],[P[X-1,:,1]],'o')                        # plotting the particles
        plt.plot(P[X - 1,:, 0], P[X - 1,:, 1])                         # line drawn between particles
        plt.xlabel('Distance along earth surface')
        plt.ylabel('Atmospheric height')
        plt.title('Refraction of light in atmosphere (Wave theory)')
        for i in range(X - 1):
            if i < X-1:
                plt.plot(Pos[i][0][0]+xplot,np.ones(len(xplot))*Pos[i][0][1],'-g')  #plotting the axes
                plt.plot(np.ones(len(yplot))*Pos[i][0][0],yplot+Pos[i][0][1],'-b')
            for j in range(3):
                plt.plot([P[i,j,0], P[i + 1,j,0]], [P[i,j,1], P[i + 1,j,1]],'-g')   #plotting the trace lines
        plt.xlim(-10,+35)
        plt.ylim(-40,+10)
        plt.pause(.1)
        plt.clf()
        if P[X-1,0,1] >= y1:                                           # Checking if particle 1 has crossed the medium and plot
            P[X -1,:,0] += dl/n1 * m.sin(theta)
            P[X -1,:,1] -= dl/n1 * m.cos(theta)
            plt.text(x1 + 4, y1 + 4, theta * 180 / m.pi, fontdict=None, withdash=False)
        if P[X-1,0,1]< y1 and P[X-1,1,1]> y1:                          # Checking if particle 2 has crossed medium after particel and plot
            P[X-1,[1,2],0] += dl/n1 * m.sin(theta)
            P[X-1,[1,2],1] -= dl/n1 * m.cos(theta)
            circle = plt.Circle((x1,y1),R1, color = 'b', fill = False)
            p1plot(circle)
            R1 += dl/n2                                                # Increment the radius of circle1
            plt.text(x1 + 4, y1 + 4, theta * 180 / m.pi, fontdict=None, withdash=False)
        if P[X-1,1,1]< y1 and P[X-1,2,1]>y1:                           # Checking if particle 3 hasn't crossed and plot
            P[X-1,2,0] += dl * m.sin(theta)
            P[X-1,2,1] -= dl * m.cos(theta)
            circlei = plt.Circle((x1,y1),R1, color = 'b', fill = False)
            circlej = plt.Circle((x2,y2),R2, color='b', fill=False)
            p2plot(circlei,circlej)
            R1 += dl/n2                                                # Increment radii of circles
            R2 += dl/n2
            plt.text(x1 + 4, y1 + 4, theta * 180 / m.pi, fontdict=None, withdash=False)
        if abs(P[X-1,2,1]-y1) <= 0.01:                                 # Checking if particle 3 has reached the interface
            circlei = plt.Circle((x1,y1), R1, color='b', fill=False)
            circlej = plt.Circle((x2,y2), R2, color='b', fill=False)
            P[X-1,0] = newton_raphson(derivfn_1, derivfn_11, M, R1, 0, 0, 0 + R1-dl/n2, 10 ** -10)  #Computing tangent points of circles
            P[X-1,1] = newton_raphson(derivfn_2, derivfn_22, M, R1/2, 0, 0, 0 + (R1-dl/n2)/4, 10 ** -10)
            refract_angle = m.atan(((y3-y1) - P[X-1,0, 1]) / ((x3-x1) - P[X-1,0, 0]))    # finding the refraction angle using slope of tangent
            P[X-1,2] = [x3 + R3 * m.sin(refract_angle), y3 - R3 * m.cos(refract_angle)]
            P[X-1,0,0] += x1
            P[X-1,0,1] += y1
            p3plot(P[X-1,0], P[X-1,1], P[X-1,2], circlei, circlej)
            R1 += dl/n2                                                # Increment radii of all circles
            R2 += dl/n2
            R3 += dl/n2
        if P[X-1,2,1]< y1:                                             # Check if particle 3 has crossed interface
            P[X-1,0] = [x1 + R1 * m.sin(refract_angle), y1 - R1 * m.cos(refract_angle)]   # particle co-ordinates
            P[X-1,1] = [x2 + R2 * m.sin(refract_angle), y2 - R2 * m.cos(refract_angle)]
            P[X-1,2] = [x3 + R3 * m.sin(refract_angle), y3 - R3 * m.cos(refract_angle)]
            R1 += dl/n2                                                # Incrementing the distance travelled in the medium
            R2 += dl/n2
            R3 += dl/n2
            plt.text(x1 + 4, y1 - 7, refract_angle * 180 / m.pi, fontdict=None, withdash=False)
    Radii.append(-dl/n2 + R1)                                          # Append the distance travelled by particle 1 to array Radii
    Refract.append(refract_angle)                                      # Append the refracted angle to array Refract
    width = width / m.cos(theta)*m.cos(refract_angle)                  # Manipulating width (it should be constant irrespective of multiple refractions)
    return [Radii, width, Pos, Refract]
#--------------------------------------------------------------------------------------------
# Initial parameters
width = 2                                                                                        # width of beam
Refract = [60 * m.pi/180]                                                                        # Incident angle
[x1,y1] = [0,0]                                                                                  # origin
Radii = [0]                                                                                      # Initial distance travelled = 0
p1_loc = [x1, y1]
p2_loc = [p1_loc[0]+width/2*m.cos(Refract[0]), p1_loc[1]+width/2*m.sin(Refract[0])]
p3_loc = [p1_loc[0]+width*m.cos(Refract[0]), p1_loc[1]+width*m.sin(Refract[0])]
Pos = [[[p1_loc[0],p1_loc[1]],[p2_loc[0],p2_loc[1]],[p3_loc[0],p3_loc[1]]]]                      # Initial locations of 3 particles
#--------------------------------------------------------------------------------------------
# Function calls

def n(h):
    return 1 + m.exp(-h/H)
h = H
dh = 1
while h > 0:
    R = refraction(n(h), n(h-1), Radii, width, Pos, Refract)
    [Radii, width, Pos, Refract] = [R[0], R[1], R[2], R[3]]
    h = h - dh


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, Checkbox, fixed

def find_points(X, y): 
    """
    Find the points A, B, C, D in X that are closest to y and store them
    """
    A = None # variable to store the point A
    B = None # variable to store the point B
    C = None # variable to store the point C
    D = None # variable to store the point D

    min_dist_A = float('inf') # variable to store the minimum distance to A
    min_dist_B = float('inf') # variable to store the minimum distance to B
    min_dist_C = float('inf') # variable to store the minimum distance to C
    min_dist_D = float('inf') # variable to store the minimum distance to D

    for x in X: # Iterate over all points in X
        dist = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2) # Calculate the distance between x and y using the Euclidean distance given in the exam question
        if x[0] > y[0] and x[1] > y[1] and dist < min_dist_A: # Check if the conditions for A are met and if the distance is less than the minimum distance to A
            A = x # Update A if the conditions are met
            min_dist_A = dist # Update the minimum distance to A
        elif x[0] > y[0] and x[1] < y[1] and dist < min_dist_B: # same procedure as above. Now we just check if the conditions for B, C or D are satisfied and update B, C and D accordingly instead if so.
            B = x
            min_dist_B = dist
        elif x[0] < y[0] and x[1] > y[1] and dist < min_dist_C:
            C = x
            min_dist_C = dist
        elif x[0] < y[0] and x[1] < y[1] and dist < min_dist_D:
            D = x
            min_dist_D = dist

    return A, B, C, D # Return the points A, B, C, D

def plot_interactive(X, y, A, B, C, D, show_random_points, show_abc, show_cda, show_points): 
    """
    Plot the points and triangles in an interactive plot
    """
    plt.figure(figsize=(8, 8))
    if show_random_points: 
        plt.scatter(X[:, 0], X[:, 1], color='blue', label='Random Points')
    plt.scatter(y[0], y[1], color='red', label='y', zorder=5)

    if show_points:
        # Plot A, B, C, D if they exist
        if A is not None: 
            plt.scatter(A[0], A[1], color='green', label='A', zorder=5)
        if B is not None:
            plt.scatter(B[0], B[1], color='purple', label='B', zorder=5)
        if C is not None:
            plt.scatter(C[0], C[1], color='orange', label='C', zorder=5)
        if D is not None:
            plt.scatter(D[0], D[1], color='brown', label='D', zorder=5)

    # Plot triangles
    if show_abc:
        if A is not None and B is not None and C is not None:
            plt.plot([A[0], B[0]], [A[1], B[1]], 'r-', label='Triangle ABC')
            plt.plot([B[0], C[0]], [B[1], C[1]], 'r-')
            plt.plot([C[0], A[0]], [C[1], A[1]], 'r-')
    if show_cda:
        if C is not None and D is not None and A is not None:
            plt.plot([C[0], D[0]], [C[1], D[1]], 'g-', label='Triangle CDA')
            plt.plot([D[0], A[0]], [D[1], A[1]], 'g-')
            plt.plot([A[0], C[0]], [A[1], C[1]], 'g-')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Points and Triangles')
    plt.grid(True)
    plt.show()

def interactive_plot(X, y, A, B, C, D):
    """
    Create the interactive plot
    """    
    interact(plot_interactive,
             X=fixed(X),
             y=fixed(y),
             A=fixed(A),
             B=fixed(B),
             C=fixed(C),
             D=fixed(D),
             show_random_points=Checkbox(value=True, description='Show Random Points'),
             show_abc=Checkbox(value=True, description='Show Triangle ABC'),
             show_cda=Checkbox(value=True, description='Show Triangle CDA'),
             show_points=Checkbox(value=True, description='Show Points A, B, C, D'))

def barycentric_coordinates(A,B,C, y):
    """
    Compute barycentric coordinates of point y with respect to an arbitrary triangle ABC.

    Returns:
    r1, r2, r3 : Barycentric coordinates of the point y with respect to triangle ABC.
    """
    A1, A2 = A # Extract coordinates of A, B and C
    B1, B2 = B 
    C1, C2 = C
    y1, y2 = y # Extract coordinates of y

    denom = (B2 - C2)*(A1 - C1) + (C1 - B1)*(A2 - C2) # Compute the denominator of the barycentric coordinates
    r1 = ((B2 - C2)*(y1 - C1) + (C1 - B1)*(y2 - C2)) / denom # Compute the barycentric coordinate r1
    r2 = ((C2 - A2)*(y1 - C1) + (A1 - C1)*(y2 - C2)) / denom # Compute the barycentric coordinate r2
    r3 = 1 - r1 - r2 # Compute the barycentric coordinate r3

    return r1, r2, r3 # Return the barycentric coordinates

def is_inside_triangle(r1, r2, r3): 
    """
    Function to check if the point is inside the triangle
    
    Returns (bool): True if the point is inside the triangle, False otherwise.
    """
    return (0 <= r1 <= 1) and (0 <= r2 <= 1) and (0 <= r3 <= 1)

def check_point_in_triangles(A, B, C, D, y):
    """
    Function to check if a point y is inside the triangles ABC and CDA specifically
    """
    # Compute barycentric coordinates for triangles ABC and CDA
    r1_ABC, r2_ABC, r3_ABC = barycentric_coordinates(A, B, C, y)
    r1_CDA, r2_CDA, r3_CDA = barycentric_coordinates(C, D, A, y)

    # Check if y is inside triangle ABC
    inside_ABC = is_inside_triangle(r1_ABC, r2_ABC, r3_ABC)

    # Check if y is inside triangle CDA
    inside_CDA = is_inside_triangle(r1_CDA, r2_CDA, r3_CDA)

    return (inside_ABC, (r1_ABC, r2_ABC, r3_ABC)), (inside_CDA, (r1_CDA, r2_CDA, r3_CDA)) # Return the results

def compute_approximation_and_true_value(f, X, y, A, B, C, D, inside_ABC, bary_coords_ABC, inside_CDA, bary_coords_CDA):
    """
    Compute the approximation of f(y) using barycentric interpolation and compare it with the true value.
    
    Parameters:
    f: function
        The function to be approximated.
    X: 
        The set of random points in the unit square.
    y: 
        The point at which to approximate the function.
    A, B, C, D : 
        Coordinates of the vertices of the triangles.
    inside_ABC : bool
        Whether the point y is inside triangle ABC.
    bary_coords_ABC:
        Barycentric coordinates of y with respect to triangle ABC.
    inside_CDA: bool
        Whether the point y is inside triangle CDA.
    bary_coords_CDA:
        Barycentric coordinates of y with respect to triangle CDA.
    
    Returns:
    f_y_true :
        The true value of the function at y.
    f_y_approx :
        The approximated value of the function at y using barycentric interpolation.
    abs_error :
        The absolute error between the true and approximated values.
    """
    # Before starting the actual algorithm we initialize the approximation of f(y). 
    # f_y_approx is set to np.nan if y is not inside any of the triangles as a placeholder value.
    f_y_approx = np.nan

    # Ensure that points A, B, C, D are not None
    if any(point is None for point in [A, B, C, D]):
        raise ValueError("One or more of the points A, B, C, D is None")

    # Step 1: Compute the function values at the vertices
    f_A = f(A)
    f_B = f(B)
    f_C = f(C)
    f_D = f(D)

    # Step 2+3: Check if y is inside ABC or CDA and compute the approximation using barycentric interpolation if y is inside one of the triangles
    if inside_ABC:
        r1, r2, r3 = bary_coords_ABC
        f_y_approx = r1 * f_A + r2 * f_B + r3 * f_C
    elif inside_CDA:
        r1, r2, r3 = bary_coords_CDA
        f_y_approx = r1 * f_C + r2 * f_D + r3 * f_A
    
    # Step 4: If y is not inside any of the triangles, we return NaN as the approximation placeholder initialized at the beginning

    # We are now finished with the algorithm and also want to compute the true value of f(y). We can use the known function f to compute this value.
    f_y_true = f(y)

    # Compute the absolute error
    abs_error = np.abs(f_y_true - f_y_approx)

    # Output the results
    print(f"True value of f({y[0]}, {y[1]}): {f_y_true:.5f}")
    print(f"Approximated value of f({y[0]}, {y[1]}): {f_y_approx:.5f}")
    print(f"Absolute error between the approximated and true value: {abs_error:.5f}")


    return f_y_true, f_y_approx, abs_error